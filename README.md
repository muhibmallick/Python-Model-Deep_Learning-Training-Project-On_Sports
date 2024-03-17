Here's a list of dependencies based on the libraries used in your script:
numpy
opencv-python
scikit-learn
imutils
tensorflow

o install these dependencies, one would typically run the following commands in the terminal:
pip install numpy
pip install opencv-python
pip install scikit-learn
pip install imutils
pip install tensorflow

Setting up Visual Studio Code:
Install Visual Studio Code:

Download and install VS Code from here.
Install the Python Extension for Visual Studio Code:

Open VS Code, go to the Extensions view by clicking on the Extensions icon on the Sidebar or pressing Ctrl+Shift+X.
Search for 'Python' and install the extension provided by Microsoft.
Open the Project Folder:

Click File > Open Folder and select the folder where your project is located.
Create a Virtual Environment:

Open the terminal in VS Code with Ctrl+` or by selecting Terminal > New Terminal from the top menu.

Inside the terminal, navigate to your project folder if not already there.

Run the following command to create a virtual environment (replace .venv with your preferred environment name):

python -m venv .venv

Activate the Virtual Environment:

For Windows, run:

.\.venv\Scripts\activate

For macOS and Linux, run:

source .venv/bin/activate

Install Dependencies in the Virtual Environment:
pip install numpy opencv-python scikit-learn imutils tensorflow


You can run the script directly in the terminal by typing:

python <script-name>.py

Script name is : main
