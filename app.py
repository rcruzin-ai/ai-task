# name: app.py
# description: main application 
# author: raymond cruzin

import streamlit as st
import altair as alt

from sentiment_analyzer import SentimentAnalyzer
from model_selector import ModelSelector
from leaderboard_scores import LeaderboardScores

def main():
    """
    Main function to run the Sentiment Analysis Web Application.
    """

    # Set the page layout to wide mode
    st.set_page_config(layout="wide")

    st.title("Sentiment Analysis Web Application")

    menu = ["ABOUT", "MAIN"]
    choice = st.sidebar.selectbox("MENU", menu)

    if choice == "MAIN":
        home_menu = ["PICK A MODEL", "LEADERBOARD"]
        home_choice = st.sidebar.selectbox("MAIN MENU", home_menu)

        if home_choice == "PICK A MODEL":
         
            st.subheader("PICK A MODEL")      

            # Create a SentimentAnalyzer object
            analyzer = SentimentAnalyzer()

            # Create a ModelSelector object
            selector = ModelSelector()

            # Create a drop-down menu in the sidebar for model selection
            model_name = st.sidebar.selectbox("Select a model:", selector.get_model_names())

            # Display the logo of the selected model
            st.sidebar.image(selector.get_model_logo(model_name), width=100)
            st.sidebar.write(model_name)

            # Insert here
            st.subheader("Option 1: Test Case")
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file:
                
                import pandas as pd
                import numpy as np
                
                test_df = pd.read_csv(uploaded_file)

                # Insert here
                index = st.number_input('Enter a row index number', min_value=0, max_value=len(test_df)-1, step=1, value=1)
                if st.button('Display Row'):                
                    st.write(test_df.iloc[index], use_container_width=True)

            st.subheader("Option 2: Enter Text")
            with st.form(key='nlpForm'):
                raw_text = st.text_area("Enter Text Here")
                submit_button = st.form_submit_button(label='Analyze')

            if submit_button:
                st.info("Results")
                
                result = analyzer.analyze_text(raw_text, selector.get_model_path(model_name))
                
                if result is not None:
                    # Convert the result dictionary to a DataFrame and display it using Streamlit            
                    result_df = analyzer.convert_to_df(result)
                    
                    # Emoji
                    max_label = result_df['score'].idxmax()
                    if max_label == 'positive':
                        st.markdown("Sentiment: Positive :smiley:")
                    elif max_label == 'negative':
                        st.markdown("Sentiment: Negative :disappointed:")
                    else:
                        st.markdown("Sentiment: Neutral :neutral_face:")
                    
                    # Visualization
                    c = alt.Chart(result_df.reset_index()).mark_bar().encode(
                        x='label',
                        y='score',
                        color='label',
                        text=alt.Text('score', format='.2f')
                    ).properties(width=600)
                    text = c.mark_text(
                        align='center',
                        baseline='middle',
                        dy=-10,
                        fontSize=20,
                        color='white'
                    ).encode(
                        text=alt.Text('score', format='.2f')
                    )
                    chart = (c + text).configure_axis(
                        labelFontSize=20,
                        titleFontSize=20
                    )
                    st.altair_chart(chart, use_container_width=True)



        elif home_choice == "LEADERBOARD":
            st.subheader("LEADERBOARD")

            # Allow users to upload multiple CSV files
            uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

            if uploaded_files:

                
                # Create a LeaderboardScores object
                csv_combiner = LeaderboardScores(uploaded_files)
                

                st.subheader("Model Prediction Vs. Test Dataset")
                # Display the first few rows of the combined DataFrame
                st.write(csv_combiner.combined_df.head())

                # Create a SentimentAnalyzer object
                sentiment_analyzer = SentimentAnalyzer()
                                
                try:                    
                    st.subheader("Evaluation Metrics for Classification Models")

                    # Compare the performance of each model using sentiment_analyzer.sentiment_mapping as input
                    results_df = csv_combiner.compare_model_performance(sentiment_analyzer.sentiment_mapping)
                                                                        
                    st.write(results_df, use_container_width=True)

                                
                    st.subheader("Weighted Ave F1_Score for Multi Classification Models")

                    f1_df, fig = csv_combiner.radar_chart_weighted_ave_f1_score(sentiment_analyzer.sentiment_mapping)

                    st.plotly_chart(fig, use_container_width=True)

                                    
                    st.write(f1_df, use_container_width=True)

                                                                                        
                except KeyError as e:
                    # Print debug information
                    print("KeyError:", e)
                    print("Combined DataFrame columns:", csv_combiner.combined_df.columns)            
                                
                                            
                try:                    
                    st.subheader("Confusion Matrix")
                    # Plot the confusion_matrix_per_model using csv_combiner.combined_df and sentiment_analyzer.sentiment_mapping as input
                    fig = csv_combiner.confusion_matrix_per_model(csv_combiner.combined_df, sentiment_analyzer.sentiment_mapping)

                    # Display the plot in Streamlit using st.plotly_chart instead of st.pyplot
                    st.plotly_chart(fig, use_container_width=True)
                                        

                except KeyError as e:
                    # Print debug information
                    print("KeyError:", e)
                    print("Combined DataFrame columns:", csv_combiner.combined_df.columns)

                                
                try:
                    st.subheader("Sentiment Class Metrics")
                    # Plot the plot_model_performance using csv_combiner.combined_df and sentiment_analyzer.sentiment_mapping as input
                    figures = csv_combiner.plot_model_performance(csv_combiner.combined_df, sentiment_analyzer.sentiment_mapping)

                    # Iterate over the list of figures
                    for fig in figures:
                        # Display the current figure using the st.plotly_chart function with full width
                        st.plotly_chart(fig, use_container_width=True)
                                        
                except KeyError as e:
                    # Print debug information
                    print("KeyError:", e)
                    print("Combined DataFrame columns:", csv_combiner.combined_df.columns)
                            
    elif choice == "ABOUT":
        
        st.subheader("ABOUT")
        
        # Read the about.md file and display it using Streamlit
        with open("docs/about.md", "r") as f:
            about_text = f.read()
            st.write(about_text)

if __name__ == '__main__':
    main()
