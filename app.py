import pandas as pd
import streamlit as st
import utils
import prompts

def main():
    # Title of the app
    st.title("GenAI in Marketing")
    chat_llm = utils.llm_chat_model()
    image_llm = utils.llm_image_model()

    # Initialize session state to track file upload and segment selection
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'selected_segment' not in st.session_state:
        st.session_state.selected_segment = None
    if 'characteristic_prompt' not in st.session_state:
        st.session_state.characteristic_prompt = ""
    if 'stable_diffusion_prompt' not in st.session_state:
        st.session_state.stable_diffusion_prompt = ""

    # Define a function to handle segment selection
    def on_segment_select():
        st.session_state.selected_segment = st.session_state.segment
        st.session_state.characteristic_prompt = ""
        st.session_state.stable_diffusion_prompt = ""

    # Upload file section
    if not st.session_state.file_uploaded:
        # File uploader
        uploaded_file = st.file_uploader("Upload File:", type=["csv", "xlsx", "json"], label_visibility="visible")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)

                df = utils.cluster_creation(df)
                st.session_state.df = df  # Store the DataFrame in session state
                st.session_state.file_uploaded = True
                st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {e}")
    else:
        # Creating columns for layout with width ratio
        left_column, right_column = st.columns([1, 3])

        # Left column: Segments dropdown
        with left_column:
            # Segments dropdown
            segments = ["None", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            st.selectbox("Segments", segments, key='segment', on_change=on_segment_select)

        # Right column: Characteristic Prompt and Stable Diffusion Prompt
        if st.session_state.selected_segment and st.session_state.selected_segment != "None":
            with right_column:
                cluster_number = st.session_state.selected_segment
                df = st.session_state.df
                cluster_data = df[df['Cluster'] == cluster_number]

                if not st.session_state.characteristic_prompt:
                    characteristic_prompt = utils.characteristic_prompt_generation(cluster_data, chat_llm)
                    st.session_state.characteristic_prompt = characteristic_prompt

                st.subheader(f"Characteristic Prompt for Segment {cluster_number}:")
                st.code(st.session_state.characteristic_prompt, language = "markdown")
                #st.text_area("Characteristic Prompt", st.session_state.characteristic_prompt, height=150)

                st.subheader(f"Stable Diffusion Prompt for Segment {cluster_number}:")
                if not st.session_state.stable_diffusion_prompt:
                    stable_diffusion_prompt = utils.stable_diffusion_prompt_generation(st.session_state.characteristic_prompt, chat_llm)
                    st.session_state.stable_diffusion_prompt = stable_diffusion_prompt

                edited_stable_diffusion_prompt = st.text_area(
                    "Stable Diffusion Prompt",
                    st.session_state.stable_diffusion_prompt,
                    height=100
                )

                if st.button("Generate Image"):
                    st.session_state.stable_diffusion_prompt = edited_stable_diffusion_prompt
                    edited_stable_diffusion_prompt += prompts.additional_image_instruction
                    #st.write(edited_stable_diffusion_prompt)
                    try:
                        image = utils.image_generator(edited_stable_diffusion_prompt, image_llm)
                        st.image(image, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error generating image: {e}")
        else:
            with right_column:
                st.write("Select a segment to see the characteristic and Stable Diffusion prompts.")

if __name__ == "__main__":
    main()
