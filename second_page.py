import streamlit as st
from chains import get_vector_store, get_retreiver_chain, get_conversational_rag
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from streamlit_feedback import streamlit_feedback
import time, uuid

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

    # One, single container owns the entire transcript UI.
    chat_area = st.container()

    def render_history():
        with chat_area:
            for message in st.session_state.chat_history:
                role = "assistant" if isinstance(message, AIMessage) else "user"
                with st.chat_message(role):
                    st.markdown(message.content)

    def get_response_streaming(user_input):
        history_retriever_chain = get_retreiver_chain(st.session_state.vector_store)
        conversation_rag_chain = get_conversational_rag(history_retriever_chain)
        for chunk in conversation_rag_chain.stream({
            "chat_history": st.session_state.chat_history,
            "input": user_input,
            "student_id": st.session_state.student_id,
            "dialog_identifier": st.session_state.dialog_identifier
        }):
            if "answer" in chunk:
                yield chunk["answer"]

    def get_response_non_streaming(user_input):
        history_retriever_chain = get_retreiver_chain(st.session_state.vector_store)
        conversation_rag_chain = get_conversational_rag(history_retriever_chain)
        resp = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input,
            "student_id": st.session_state.student_id,
            "dialog_identifier": st.session_state.dialog_identifier
        })
        return resp["answer"]

    # First, render existing history.
    render_history()

    if user_input := st.chat_input("Type your message here..."):
        # 1) Persist and 2) IMMEDIATELY render the new user message.
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with chat_area:
            with st.chat_message("user"):
                st.markdown(user_input)

        with collect_runs() as cb:
            # Stream the assistant reply inside the same container.
            with chat_area:
                with st.chat_message("assistant"):
                    status = st.status("Working...", expanded=True)
                    status.write("🔎 Retrieving context…")
                    t0 = time.time()
                    time.sleep(0.05)
                    status.write("🧠 Reasoning…")

                    message_placeholder = st.empty()
                    full_response = ""
                    got_first_chunk = False

                    try:
                        for chunk in get_response_streaming(user_input):
                            if not got_first_chunk:
                                status.update(
                                    label=f"🧩 Thought for {time.time()-t0:.1f}s, now generating…",
                                    state="running", expanded=True
                                )
                                status.write("✍️ Generating answer…")
                                got_first_chunk = True
                            if chunk:
                                full_response += chunk
                                message_placeholder.markdown(full_response + "│")
                                # tiny delay helps visually but isn't required
                                time.sleep(0.01)

                        if not got_first_chunk:
                            status.update(label="⚠️ No streamed chunks; using fallback.", state="error")
                            full_response = get_response_non_streaming(user_input)
                            message_placeholder.markdown(full_response)

                        message_placeholder.markdown(full_response)
                        status.update(label="✅ Done", state="complete", expanded=False)

                    except Exception as e:
                        status.update(label="⚠️ Streaming failed; using fallback.", state="error")
                        st.error(f"Streaming failed. Error: {e}")
                        full_response = get_response_non_streaming(user_input)
                        message_placeholder.markdown(full_response)

        # Persist assistant message, then clean rerender to remove placeholders.
        st.session_state.chat_history.append(AIMessage(content=full_response))
        st.session_state.run_id = cb.traced_runs[0].id
        st.rerun()

    # Feedback (unchanged)
    feedback_option = "thumbs"
    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            optional_text_label="[Optional] Please provide an explanation",
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
                rec = client.create_feedback(
                    run_id, feedback_type_str, score=score, comment=feedback.get("text"),
                )
                st.session_state.feedback = {"feedback_id": str(rec.id), "score": score}
            else:
                st.warning("Invalid feedback score.")
