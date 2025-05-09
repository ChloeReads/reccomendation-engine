############ Import Dependencies ################

import tkinter as tk  # Importing the Tkinter library for GUI development
from pyuiWidgets.listBox import ScrollableListbox  # Custom widget for scrollable listbox
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting data
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Tkinter integration for Matplotlib
from matplotlib.figure import Figure  # Importing the Figure class for chart creation
import pandas as pd  # Importing Pandas for data manipulation
from sklearn.feature_extraction.text import CountVectorizer  # This will be used for converting descriptor text of books into Vectors
from sklearn.metrics.pairwise import cosine_similarity  # Computing similarities between books

############### Data Import and Processing #####################

# Load dataset from CSV file
df_source = pd.read_csv('data/ReccomendationEngineSource.csv')

# Remove duplicate entries based on the 'Title' column
df_cleaned = df_source.drop_duplicates(subset='Title')

# Convert text data to lowercase for as a requirement for CountVectorizer function
df_cleaned['Title'] = df_cleaned['Title'].str.lower()
df_cleaned['Author'] = df_cleaned['Author'].str.lower()
df_cleaned['Narrator'] = df_cleaned['Narrator'].str.lower()
df_cleaned['Series'] = df_cleaned['Series'].str.lower()
df_cleaned['Genres'] = df_cleaned['Genres'].str.lower()

# Convert text to lower case and remove new line characters to clean up input
df_cleaned['Description'] = df_cleaned['Description'].str.lower().replace(to_replace="\n",value=" ").replace(to_replace="\r",value=" ")

# Sort list Alphabetically
df_cleaned.sort_values(by=['Title'], inplace=True)

# Combine relevant columns into a single column for similarity analysis
df_combined = df_cleaned.drop(['ReadableTitle'],axis=1)
df_combined['data'] = df_combined[df_combined.columns[1:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),  # Concatenating non-null text values
    axis=1
)

df_combined['ReadableTitle'] = df_cleaned['ReadableTitle']

############# Build our Cosine Similarity Model ################

# Convert text data into a matrix of token counts
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df_combined['data'])

# Compute cosine similarity matrix
similarities = cosine_similarity(vectorized)

# Store similarity data in a DataFrame for easy retrieval
df_similarities = pd.DataFrame(similarities, columns=df_cleaned['ReadableTitle'],index=df_cleaned['ReadableTitle']).reset_index()

############### Create and Layout Main Window ###############

# Define the main window for the application
main = tk.Tk()
main.config(bg="#939393")  # Set background color
main.title("ZacLib Recommender")  # Window title
main.geometry("800x500")  # Set window dimensions
main.iconbitmap(default="Img/Audiobook_Square.ico")

# Define label for the application title
label = tk.Label(master=main, text="ZacLib - Audiobook Recommender")
label.config(bg="#E4E2E2", fg="#000", font=("Helvetica", 32))
label.place(x=10, y=10, width=780, height=50)

# Define frames for different sections of the UI
frame = tk.Frame(master=main)  # Frame for selection menu
frame.config(bg="#EDECEC")
frame.place(x=10, y=70, width=320, height=150)

frame1 = tk.Frame(master=main)  # Frame for similarity chart
frame1.config(bg="#EDECEC")
frame1.place(x=10, y=230, width=780, height=260)

frame2 = tk.Frame(master=main)  # Frame for recommendations list
frame2.config(bg="#EDECEC")
frame2.place(x=340, y=70, width=450, height=150)

############ Build our Interactables ###################

# Create dropdown menu for book selection
option_menu_options = df_similarities['ReadableTitle']
option_menu_var = tk.StringVar(value="Select option")  # Default placeholder
option_menu = tk.OptionMenu(frame, option_menu_var, *option_menu_options)
option_menu.config(bg="#E4E2E2", fg="#000")
option_menu.place(x=10, y=10, width=300, height=40)

# Label for displaying when no recommendations are found
no_recs = tk.Label(master=frame2, text="No Similar Books Found")
no_recs.config(bg="#E4E2E2", fg="#000", font=("Helvetica", 16))

# Scrollable list box to display recommendations
list_box = ScrollableListbox(parent=frame2, scrollx=False, scrolly=True)
list_box.config(bg="#E4E2E2", fg="#000")

# Function to generate recommendations and display similarity chart
def output_recs():
    # Hide previous recommendations and reset list box
    no_recs.place_forget()
    list_box.place_forget()
    list_box.delete(0, tk.END)
    
	# Pull input_book from the selected book in the book selection dropdown
    input_book = option_menu_var.get()

    # Find the top 6 most similar books
    recommendations = pd.DataFrame(df_similarities.nlargest(6, input_book)).sort_values(by=input_book, ascending=True)
    recommendations = recommendations[recommendations['ReadableTitle'] != input_book]
    recommendations = recommendations[recommendations[input_book] > 0]

    # Create a bar chart to visualize similarity scores
    fig = Figure(figsize=(2, 2), dpi=100, constrained_layout=True)
    fig.patch.set_facecolor("#E4E2E2")
    ax = fig.add_subplot(111)
    charts = FigureCanvasTkAgg(fig, master=frame1)

    labels = recommendations['ReadableTitle']
    values = recommendations[input_book]
    ax.barh(labels, values, color="#8c97e6")
    ax.set_title("Book Similarity")
    ax.set_facecolor("#E4E2E2")

    charts.draw()
    charts.get_tk_widget().place(x=10, y=10, width=760, height=240)

    # Display recommendations list if similar books are found
    if recommendations[input_book].sum() != 0:
        for i in recommendations['ReadableTitle']:
            list_box.insert(tk.END, i)
        list_box.place(x=10, y=10, height=130, width=430)
    else:
        for i in frame1.winfo_children():
            i.destroy()
        no_recs.place(x=10, y=10, width=430, height=130)

# Button to trigger recommendation generation
generate = tk.Button(master=frame, text="Give me Recommendations!", command=lambda: output_recs())
generate.config(bg="#E4E2E2", fg="#000")
generate.place(x=76, y=81, width=167, height=49)

# Required for UI functionality
main.mainloop()
