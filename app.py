import pandas as pd
import streamlit as st
import util
import prompts

def main():
    # Set custom page configuration
    st.set_page_config(page_title="GenAI in Marketing", page_icon=":chart_with_upwards_trend:", layout="wide")

    # Title of the app with custom markdown styling
    st.markdown(
        """
        <div style="background-color:#00008B;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">GenAI in Marketing</h1>
        </div>
        """, unsafe_allow_html=True
    )

    # Initialize chat and image models
    chat_llm = util.llm_chat_model()
    image_llm = util.llm_image_model()

    # Initialize session state
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
        # File uploader with progress bar
        st.markdown("### Upload Your Data File")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"], label_visibility="visible")
        if uploaded_file is not None:
            with st.spinner('Processing file...'):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        df = pd.read_json(uploaded_file)
                    df = util.cluster_creation(df)
                    st.session_state.df = df  # Store the DataFrame in session state
                    st.session_state.file_uploaded = True
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error processing file: {e}")
    else:
        # Creating columns for layout with width ratio
        st.markdown("---")
        col1,col2,col3,col4,col5 = st.columns([0.75, 0.05,0.05, 3, 3])

        # Left column: Segments dropdown and text areas
        with col1:
            # Segments dropdown with header and icon
            st.markdown("### Select a Segment :bar_chart:")
            segments = ["None", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            st.selectbox("Segments", segments, key='segment', on_change=on_segment_select)
            if st.session_state.selected_segment and st.session_state.selected_segment != "None":
                cluster_number = st.session_state.selected_segment
                df = st.session_state.df
                cluster_data = df[df['Cluster'] == cluster_number]

                if not st.session_state.characteristic_prompt:
                    characteristic_prompt = util.characteristic_prompt_generation(cluster_data, chat_llm)
                    st.session_state.characteristic_prompt = characteristic_prompt

                with col2:
                    st.markdown(
                """
                <style>
                .vertical-line {
                    border-left: 5px solid #00008B;
                    height: 150vh;
                    position: absolute;
                }
                </style>
                <div class="vertical-line"></div>
                """,
                unsafe_allow_html=True
            )

                with col4:
                    st.subheader(f"Characteristic Prompt for Segment {cluster_number}:")
                    st.code(st.session_state.characteristic_prompt, language="markdown")

                    st.subheader(f"Stable Diffusion Prompt for Segment {cluster_number}:")
                    if not st.session_state.stable_diffusion_prompt:
                        stable_diffusion_prompt = util.stable_diffusion_prompt_generation(st.session_state.characteristic_prompt, chat_llm)
                        st.session_state.stable_diffusion_prompt = stable_diffusion_prompt

                    edited_stable_diffusion_prompt = st.text_area(
                        "Stable Diffusion Prompt",
                        st.session_state.stable_diffusion_prompt,
                        height=100
                    )

                with col5:
                    if st.button("Generate Image", help="Click to generate an image based on the edited prompt"):
                        st.session_state.stable_diffusion_prompt = edited_stable_diffusion_prompt
                        edited_stable_diffusion_prompt += prompts.additional_image_instruction
                        try:
                            with st.spinner('Generating image...'):
                                image = util.image_generator(edited_stable_diffusion_prompt, image_llm)
                                st.image(image, use_column_width=True)
                        except Exception as e:
                            st.error(f"Error generating image: {e}")
            else:
                st.write("Select a segment to see the characteristic and Stable Diffusion prompts.")


if __name__ == "__main__":
    main()
