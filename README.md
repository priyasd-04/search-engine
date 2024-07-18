# CS121-Assignment3-M1
Milestone 1: Index construction
CS121-Assignment3-M1 : Search Engine
Names: Shavonne Lin, Wilberth Gonzalez, Shayan Islam, Priya Deshmukh


Indexer.py
This file builds the inverted index run on the DEV folder which is later used in the main search engine. 

To set up for usage:
Ensure all proper python libraries are downloaded
Navigate to the cloned repository or file of the project
The proper files and folders are downloaded and able to be navigated to

To build an index
First define the folder paths to the DEV folder, along with the index folder and url map path
Create an instance of the Indexer with these three parameters being called in it
call .build_index on the instance

Example:
dev_folder_path = "<path_to_DEV>"
index_folder_path = "./index"
url_map_path = "./index/urls.pkl"

indexer = Indexer(dev_folder_path, index_folder_path, url_map_path)
indexer.build_index()

Search.py:
This file uses the built index and pkl files in various ways to return URLs with words in the search query. It utilizes ranking methods to give the most relevant results for the user.

To run Search.py:
You should already have your index built from the previous steps which will give you the index and url map folder path
Define the paths of interest
Create a search engine instance with the parameters and begin the search

Example:
index_folder_path = "./index"
urls_pkl_path = "./index/urls.pkl"

search_engine = SearchEngine(index_folder_path, urls_pkl_path)

search_engine.start()

App.py: This runs the search engine in the same way but in the form of a local GUI
To run this application, just launch the app.py file or click the run button on your IDE

A new window should appear prompting for a search query.
Type in any search querys you would like and click on the linked results to open the pages.
