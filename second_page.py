import streamlit as st
from chains import get_vector_store, get_retreiver_chain, get_conversational_rag
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from utils import load_docs_from_jsonl
from langchain_community.document_loaders.csv_loader import CSVLoader

import time
import uuid

client = Client()


def second_page():
    st.header("AI504 Chatbot")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Go to Home", key="home_page"):
            st.session_state.pop("student_id", None)
            st.session_state.pop("chat_history", None)
            st.session_state.pop("dialog_identifier", None)
            st.rerun()

    with col2:
        if st.button("Refresh", key="refresh"):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("dialog_identifier", None)
            st.rerun()


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store()
    if "dialog_identifier" not in st.session_state:
        st.session_state.dialog_identifier = uuid.uuid4()
    # if "doc" not in st.session_state:
    #     st.session_state.docs = load_docs_from_jsonl("docs/doc.jsonl")


    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        else:
            with st.chat_message("Human"):
                st.write(message.content)


    def get_response(user_input):
        history_retriever_chain = get_retreiver_chain(st.session_state.vector_store)
        conversation_rag_chain = get_conversational_rag(history_retriever_chain)
        response = conversation_rag_chain.invoke({
            "chat_history":st.session_state.chat_history,
            "input":user_input,
            "student_id" : st.session_state.student_id,
            "dialog_identifier" : st.session_state.dialog_identifier
        })
        return response["answer"]
    
    def get_response_streaming(user_input):
        """
        Streaming response function that yields chunks of the response
        """
        history_retriever_chain = get_retreiver_chain(st.session_state.vector_store)
        conversation_rag_chain = get_conversational_rag(history_retriever_chain)
        
        # Stream the response
        for chunk in conversation_rag_chain.stream({
            "chat_history": st.session_state.chat_history,
            "input": user_input,
            "student_id": st.session_state.student_id,
            "dialog_identifier": st.session_state.dialog_identifier
        }):
            if "answer" in chunk:
                yield chunk["answer"]

    def get_response_non_streaming(user_input):
        """
        Non-streaming version for comparison/fallback
        """
        history_retriever_chain = get_retreiver_chain(st.session_state.vector_store)
        conversation_rag_chain = get_conversational_rag(history_retriever_chain)
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input,
            "student_id": st.session_state.student_id,
            "dialog_identifier": st.session_state.dialog_identifier
        })
        return response["answer"]

    if user_input := st.chat_input("Type your message here..."):
        st.chat_message("Human").write(f"{user_input}")
        
        with collect_runs() as cb:
            
            with st.chat_message("AI"):
                # --- Status panel to show phases BEFORE streaming begins ---
                status = st.status("Working...", expanded=True)
                step_retr = status.write("🔎 Retrieving context…")
                # Start a timer to measure “thinking time”
                t0 = time.time()

                time.sleep(0.1)
                step_reason = status.write("🧠 Reasoning…")
                message_placeholder = st.empty()

                full_response = ""
                first_chunk_received = False

                try:
                    # Try streaming; the first token may take time to arrive.
                    for chunk in get_response_streaming(user_input):
                        if not first_chunk_received:
                            # We just transitioned from “thinking” to “generating”.
                            think_secs = time.time() - t0
                            status.update(
                                label=f"🧩 Thought for {think_secs:.1f}s, now generating…",
                                state="running",
                                expanded=True
                            )
                            status.write("✍️ Generating answer…")
                            first_chunk_received = True

                        if chunk:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "│")
                            # Optional tiny delay to make the cursor visible
                            time.sleep(0.02)

                    # If we never got a chunk, we still want to say something
                    if not first_chunk_received:
                        status.update(label="⚠️ No streamed chunks received; falling back.", state="error")
                        full_response = get_response_non_streaming(user_input)
                        message_placeholder.markdown(full_response)

                    # Finalize UI
                    message_placeholder.markdown(full_response)
                    status.update(label="✅ Done", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="⚠️ Streaming failed; using fallback.", state="error")
                    st.error(f"Streaming failed, falling back to standard response. Error: {str(e)}")
                    full_response = get_response_non_streaming(user_input)
                    message_placeholder.markdown(full_response)
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=full_response))
            st.session_state.run_id = cb.traced_runs[0].id


    feedback_option = "thumbs"
    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id
        feedback = streamlit_feedback(
            feedback_type = "thumbs",
            optional_text_label ="[Optional] Please provide an explanation",
            key=f"feedback_{run_id}",
        )

        score_mappings = {
            "thumbs": {"👍": 1, "👎": -1},
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        scores = score_mappings[feedback_option]

        if feedback:
            score = scores.get(feedback["score"])

            if score is not None:
                feedback_type_str = f"{feedback_option} {feedback['score']}"

                feedback_record = client.create_feedback(
                    run_id,
                    feedback_type_str,
                    score = score,
                    comment=feedback.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
            else:
                st.warning("Invalid feedback score.")