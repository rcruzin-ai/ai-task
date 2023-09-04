import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


class LeaderboardScores:

    def __init__(self, files):
        self.files = files
        self.dataframes = [pd.read_csv(file) for file in files]
        self.combined_df = pd.concat(self.dataframes, axis=1)
        self.combined_df = self.combined_df.drop(["text", "expected_sentiment"], axis=1)
        self.model_names = [file.name.replace("output_", "").replace(".csv", "") for file in files]
        self.column_names = []
        for model_name in self.model_names:
            self.column_names.extend([f"{model_name}_model_output", f"{model_name}_confidence_score"])
        self.combined_df.columns = self.column_names
        self.combined_df.insert(0, "text", self.dataframes[0]["text"])
        self.combined_df.insert(1, "expected_sentiment", self.dataframes[0]["expected_sentiment"])
    
    def compare_model_performance(self, sentiment_mapping):

        # Create an empty DataFrame to store the results
        results_df = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1_score"], index=self.model_names)
        
        # Iterate over the list of models
        for model_name in self.model_names:
            # Extract the expected sentiment and model output columns from the combined DataFrame
            output_df = self.combined_df[["expected_sentiment", f"{model_name}_model_output"]].copy()
            
            # Map the model output labels to a common format
            output_df[f"{model_name}_model_output"] = output_df[f"{model_name}_model_output"].apply(lambda x: next((k for k, v in sentiment_mapping.items() if x in v), x))
            
            # print(output_df.head())

            # Calculate the accuracy
            accuracy = accuracy_score(output_df["expected_sentiment"], output_df[f"{model_name}_model_output"])
            
            # Calculate the precision
            precision = precision_score(output_df["expected_sentiment"], output_df[f"{model_name}_model_output"], average="weighted")
            
            # Calculate the recall
            recall = recall_score(output_df["expected_sentiment"], output_df[f"{model_name}_model_output"], average="weighted")
            
            # Calculate the F1 score
            f1 = f1_score(output_df["expected_sentiment"], output_df[f"{model_name}_model_output"], average="weighted")
                        
            # Store the formatted results in the results DataFrame
            results_df.loc[model_name] = [accuracy, precision, recall, f1]


        results_df = (results_df*100).round(2)

        # Return the results DataFrame
        return results_df.sort_values(by='accuracy', ascending=False)
    
    def radar_chart_weighted_ave_f1_score(self, sentiment_mapping):
    
        
        # Create an empty DataFrame to store the results
        results_df = pd.DataFrame(columns=["model", "negative", "neutral", "positive"])
        
        # Iterate over the list of models
        for model_name in self.model_names:

            # Extract the expected sentiment and model output columns from the combined DataFrame
            output_df = self.combined_df[["expected_sentiment", f"{model_name}_model_output"]].copy()
            
            # Map the model output labels to a common format
            output_df[f"{model_name}_model_output"] = output_df[f"{model_name}_model_output"].apply(lambda x: next((k for k, v in sentiment_mapping.items() if x in v), x))
            
            # Group the dataframe by unique values of column 'B'
            grouped_df = output_df.groupby('expected_sentiment')
            
            # Calculate the F1 score per group class
            f1_scores = [model_name]
            for name, group in grouped_df:
                f1_scores.append(f1_score(group["expected_sentiment"], group[f"{model_name}_model_output"], average="weighted"))

            # Append the results to the DataFrame
            results_df.loc[len(results_df)] = f1_scores

        # Display the keys of the DataFrame
        # Transpose the DataFrame and set the new index
        
        transposed_df = results_df.set_index('model').T
        transposed_df = transposed_df.applymap(lambda x: round(x * 100, 2))
        
        
        # Create radar chart
        fig = go.Figure()

        for index, row in transposed_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row.values,
                theta=transposed_df.columns,
                name=index,
                mode='markers'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[0, 20, 40, 60, 80, 100]
                )),
            showlegend=True
        )

        results_df = results_df.apply(lambda row: row * 100 if row.name != 'model' else row).round(2)

        return results_df,fig
    
    def plot_model_performance(self, df, sentiment_mapping, mismatch_only=False):

        # Create a copy of the input DataFrame
        plot_df = df.copy()
        
        # Map the expected sentiment labels to a common format
        plot_df["expected_sentiment"] = plot_df["expected_sentiment"].apply(lambda x: next((k for k, v in sentiment_mapping.items() if x in v), x))
        
        # Create a list to store the figures for each sentiment class
        figures = []
        
        # Iterate over the list of sentiment classes
        for sentiment_class in ["positive", "negative", "neutral"]:
            # Filter all rows where the expected sentiment is equal to the current sentiment class
            filtered_df = plot_df[plot_df["expected_sentiment"] == sentiment_class]
            
            # If mismatch_only is True, filter only the rows where the expected sentiment is not equal to any of the model outputs
            if mismatch_only:
                filtered_df = filtered_df[~filtered_df.apply(lambda row: any(row[f"{model_name}_model_output"].apply(lambda x: next((k for k, v in sentiment_mapping.items() if x in v), x)) == row["expected_sentiment"] for model_name in self.model_names), axis=1)]
            
            # Create a figure
            fig = go.Figure()
            
            # Iterate over the list of models
            for model_name in self.model_names:
                # Add a scatter trace for the current model output
                fig.add_trace(go.Scatter(
                    x=filtered_df.index,
                    y=filtered_df[f"{model_name}_confidence_score"],
                    mode="markers",
                    name=model_name,
                    text=filtered_df["text"],  # This will be displayed when hovering over the data point
                    marker=dict(
                        color=np.where(filtered_df["expected_sentiment"] == filtered_df[f"{model_name}_model_output"].apply(lambda x: next((k for k, v in sentiment_mapping.items() if x in v), x)), "green", "red")
                    )
                ))
            
            # Update the layout of the figure
            fig.update_layout(
                title=f"{sentiment_class.capitalize()} Sentiment Class",
                xaxis_title="Text Index",
                yaxis_title="Confidence Score",
                hovermode="closest"  # This will display the hover text for the closest data point
            )
            
            # Add the figure to the list of figures
            figures.append(fig)
        
        # Return the list of Plotly figure objects
        return figures

    def confusion_matrix_per_model(self, df, sentiment_mapping):
        # Create a copy of the input DataFrame
        plot_df = df.copy()
        
        # Map the expected sentiment labels to a common format
        plot_df["expected_sentiment"] = plot_df["expected_sentiment"].apply(lambda x: next((k for k, v in sentiment_mapping.items() if x in v), x))
        
        # Create a list to store the data for each model
        data = []
        
        # Iterate over the list of models
        for model_name in self.model_names:
            # Extract the expected sentiment and model output columns for the current model
            model_df = plot_df[["text", "expected_sentiment", f"{model_name}_model_output"]].copy()
            
            # Map the model output labels to a common format
            model_df[f"{model_name}_model_output"] = model_df[f"{model_name}_model_output"].apply(lambda x: next((k for k, v in sentiment_mapping.items() if x in v), x))
            
            # Rename the columns
            model_df.columns = ["Text", "Expected Sentiment", "Model Output"]
            
            # Add a column with the model name
            model_df["Model"] = model_name
            
            # Append the data for the current model to the list
            data.append(model_df)
        
        # Concatenate the data for all models into a single DataFrame
        plot_df = pd.concat(data)
        
        # Create a figure with subplots for each model
        fig = make_subplots(rows=1, cols=len(self.model_names), subplot_titles=self.model_names)
        
        annotations = []
        
        # Iterate over the list of models and axes
        for i, model_name in enumerate(self.model_names):
            # Create a confusion matrix for the current model
            confusion_matrix = pd.crosstab(plot_df[plot_df["Model"] == model_name]["Expected Sentiment"], plot_df[plot_df["Model"] == model_name]["Model Output"], rownames=["Expected"], colnames=["Predicted"])
            
            # Plot the confusion matrix as a heatmap on the current axis
            fig.add_trace(
                go.Heatmap(
                    z=confusion_matrix.values,
                    x=confusion_matrix.columns,
                    y=confusion_matrix.index,
                    colorscale='Blues',
                    showscale=False,
                    text=confusion_matrix.values,
                    hovertemplate='Predicted: %{x}<br>Expected: %{y}<br>Count: %{z}<extra></extra>'),
                row=1,
                col=i+1)
            
            for n, row in enumerate(confusion_matrix.values):
                for m, val in enumerate(row):
                    annotations.append(
                        dict(
                            text=str(val),
                            x=confusion_matrix.columns[m],
                            y=confusion_matrix.index[n],
                            xref='x' + str(i + 1),
                            yref='y' + str(i + 1),
                            showarrow=False,
                            font=dict(color='black' if val/confusion_matrix.values.max() < 0.5 else 'white')
                        )
                    )
                    
            fig.update_xaxes(title_text="Predicted", row=1, col=i+1)
            fig.update_yaxes(title_text="Expected", row=1, col=i+1)
                    
        fig.update_layout(annotations=annotations)            
        
        # Return the Plotly figure object
        return fig
