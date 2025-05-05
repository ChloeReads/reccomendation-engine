############ Import Dependancies ################

import tkinter as tk
from pyuiWidgets.listBox import ScrollableListbox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

############### Data Import and Processing #####################

df_source = pd.read_csv('ReccomendationEngineSource.csv')
df_cleaned = df_source.drop_duplicates(subset='Title')
df_cleaned['Title'] = df_cleaned['Title'].str.lower()
df_cleaned['Author'] = df_cleaned['Author'].str.lower()
df_cleaned['Narrator'] = df_cleaned['Narrator'].str.lower()
df_cleaned['Series'] = df_cleaned['Series'].str.lower()
df_cleaned['Genres'] = df_cleaned['Genres'].str.lower()
df_cleaned['Description'] = df_cleaned['Description'].str.lower()

df_combined = df_cleaned
df_combined['data'] = df_combined[df_combined.columns[1:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)

############# Build our Cosine Similarity Model ################

vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df_combined['data'])
similarities = cosine_similarity(vectorized)
df_similarities = pd.DataFrame(similarities, columns=df_cleaned['ReadableTitle'],index=df_cleaned['ReadableTitle']).reset_index()

############### Create and Layout Main Window ###############

# Define the main window
main = tk.Tk()
main.config(bg="#939393")
main.title("ZacLib Recommender")
main.geometry("800x500")

# Define Title Element
label = tk.Label(master=main, text="ZacLib - Audiobook Recommender")
label.config(bg="#E4E2E2", fg="#000", font=("Helvetica", 32, ))
label.place(x=10, y=10, width=780, height=50)

# Define First Frame to hold Selection Element
frame = tk.Frame(master=main)
frame.config(bg="#EDECEC")
frame.place(x=10, y=70, width=320, height=150)

# Define Second Frame to hold Chart Element
frame1 = tk.Frame(master=main)
frame1.config(bg="#EDECEC")
frame1.place(x=10, y=230, width=780, height=260)

# Define Third frasme to hold Recommendations List
frame2 = tk.Frame(master=main)
frame2.config(bg="#EDECEC")
frame2.place(x=340, y=70, width=450, height=150)

############ Build our Interactables ###################

# Create Dropdown Option Menu (allows user to select Book)
option_menu_options = df_similarities['ReadableTitle']
option_menu_var = tk.StringVar(value="Select option")
option_menu = tk.OptionMenu(frame, option_menu_var, *option_menu_options)
option_menu.config(bg="#E4E2E2", fg="#000")
option_menu.place(x=10, y=10, width=300, height=40)

no_recs = tk.Label(master=frame2, text="No Similar Books Found")
no_recs.config(bg="#E4E2E2", fg="#000", font=("Helvetica", 16))
list_box = ScrollableListbox(parent=frame2, scrollx=False, scrolly=True)
list_box.config(bg="#E4E2E2", fg="#000")

# Variable to create and populate recommendations list and similarities chart
def output_recs():

	no_recs.place_forget()
	list_box.place_forget()
	list_box.delete(0,tk.END)
	input_book = option_menu_var.get()

	recommendations = pd.DataFrame(df_similarities.nlargest(6,input_book)).sort_values(by=input_book,ascending=True)
	recommendations = recommendations[recommendations['ReadableTitle']!=input_book]
	recommendations = recommendations[recommendations[input_book] > 0]

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
	
	if recommendations[input_book].sum() != 0:
		for i in recommendations['ReadableTitle']:
			list_box.insert(tk.END, i)
		list_box.place(x=10, y=10, height=130, width= 430)
	else:	
		for i in frame1.winfo_children():
			i.destroy()
		no_recs.place(x=10, y=10, width=430, height=130)

# Button used to call the above variable
generate = tk.Button(master=frame, text="Give me Recommendations!", command=lambda: output_recs())
generate.config(bg="#E4E2E2", fg="#000")
generate.place(x=76, y=81, width=167, height=49)

# required for UI functionality
main.mainloop()