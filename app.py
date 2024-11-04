import streamlit as st
from PIL import Image
import user_codes.user_defined_functions as udf

def main():

    logo = Image.open("youtube_logo.png")
    resized_logo = logo.resize((500, 150))
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(resized_logo)
    # Set the title with CSS styling for centering
    st.markdown(
        "<h1 style='text-align:center;'>Youtube Assistant - AI</h1>",
        unsafe_allow_html=True
    )

    video_url = st.text_input(label = "What is the Youtube video URL", max_chars=50)
    query = st.text_input(label = "Ask about the video content like what is the topic discussed in this link")
    submit = st.button("Submit")

    if  submit:
        
        db = udf.create_vector_db_from_youtube_url(video_url)
        response = udf.get_response_from_query(db, query, k=4)

        st.video(video_url)
        st.markdown(response)


if __name__ == "__main__":
    main()