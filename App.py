import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import openai
import os
import requests
import asyncio
from Search import SearchEngine


class CustomLineEdit(QtWidgets.QLineEdit):
    # triggers search process for enter button
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.parent().on_search()  
        super().keyPressEvent(event)

class SearchApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        openai.api_key = os.getenv('sk-proj-A4qHv7ExndBq6c2T353CT3BlbkFJcy6yV97rMFZvTdKSJdOQ')
        self.search_engine = SearchEngine(index_folder_path=r".\index", urls_pkl_path=r".\index\urls.pkl")

    def initUI(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        self.title = QtWidgets.QTextBrowser(self)
        self.title.setHtml("<h1>Search Engine</h1>")
        self.title.setFixedHeight(60)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setReadOnly(True)
        layout.addWidget(self.title)

        # Search bar
        self.search_bar = CustomLineEdit(self)
        self.search_bar.setPlaceholderText("Enter a search query")
        layout.addWidget(self.search_bar)

        # button for search
        search_button = QtWidgets.QPushButton('Search', self)
        search_button.clicked.connect(self.on_search)
        layout.addWidget(search_button)

        self.setWindowTitle('CS 121 Search Engine')
        self.setGeometry(300, 300, 600, 400)

        self.results = QtWidgets.QTextBrowser(self)
        self.results.setOpenExternalLinks(True)
        layout.addWidget(self.results)

    async def summarize_text(self, url):
        try:
            response = await asyncio.to_thread(requests.get, url)
            content = response.text

            prompt = f"Summarize: {content[:4000]}"  # Limiting to first 4000 characters

            # Using the OpenAI API to generate the summary
            completion = await openai.Completion.create(
                engine="davinci-codex",  # Or choose another engine
                prompt=prompt,
                max_tokens=150
            )

            summary = completion.choices[0].text.strip()
            return summary
        except Exception as e:
            return f"Failed to fetch summary: {str(e)}"


    async def on_search_async(self, query):
        print("Query:", query)
        self.results.clear()
        results = []
        if self.search_engine.status == "success":
            results = self.search_engine.search(query)[:20]  # Limiting to the first 20 results
            if not results:
                self.results.append("No results found.")
                return
        else:
            self.results.append("Search Engine - Index not created.")
            return
         # if summary doesnt work
        for i, (url, score) in enumerate(results, start=1):
            self.results.append(f"{i}. <a href='{url}'>{url}</a><br><br>")

        """
                tasks = [self.summarize_text(url) for url, _ in results]
        summaries = await asyncio.gather(*tasks)

        for i, (url, score) in enumerate(results, start=1):
            summary = summaries[i-1] if i-1 < len(summaries) else "Summary not available."
            self.results.append(f"{i}. <a href='{url}'>{url}</a><br>Summary: {summary}<br><br>")
        """


    def on_search(self):
        query = self.search_bar.text()
        self.search_bar.clear()
        asyncio.run(self.on_search_async(query))

    # exit
    def closeEvent(self, event: QtGui.QCloseEvent):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle('Message')
        msg_box.setText("Are you sure you want to quit?")

        yes_button = msg_box.addButton('Yes', QtWidgets.QMessageBox.YesRole)
        no_button = msg_box.addButton('No', QtWidgets.QMessageBox.NoRole)
        
        msg_box.exec_()

        if msg_box.clickedButton() == yes_button:
            print("Accepted close")
            event.accept()
            app_instance = QtWidgets.QApplication.instance().quit()
            if app_instance is not None:
                QtWidgets.QApplication.quit()
        else:
            print("Ignored close")
            event.ignore()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = SearchApp()
    ex.show()
    try:
        sys.exit(app.exec_())
    except SystemExit as e:
        print("Exiting: ", e)
