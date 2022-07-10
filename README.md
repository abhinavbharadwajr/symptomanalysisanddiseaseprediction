# Symptom Analysis and Disease Prediction

***Exploring Various Machine Learning Algorithms to possibily predict the Disease of the Patient based on their Symptoms Reported.***

# DataSets

***Final Working Data set is sourced from another GitCommit. You can find them in [Data Set (Main)](https://github.com/abhinavbharadwajr/symptomanalysisanddiseaseprdiction/tree/master/Data%20Set%20(Main)) folder.***

***Also there are some other Data Sets that were used as References for this Project. You can find them in [Data Set (Referred)](https://github.com/abhinavbharadwajr/symptomanalysisanddiseaseprdiction/tree/master/Data%20Set%20(Refered)) folder***

# Requierments for the Project

### **Anaconda Suite**

***Download the Anaconda Distribution for your Working Platform (Windows / Linux / macOS) from [here](https://www.anaconda.com/products/distribution#Downloads)***

### **Visual Studio / Visual Studio Code**

***Alternatively, you can use any other Code Editor (recommended Visual Studio Code) - but make sure you have the below listed Python Modules installed***

***Visual Studio Code make use of integrated python Terminal - which is similar to `pip`. So, you can install the packages with command like :***
```python
pip install <package-name>
```

<details><summary> Numpy </summary>
<p>

***[Numpy](https://numpy.org/) is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.***
</p>
</details>

***Install `numpy` by :***
```python
pip install numpy
```

<details><summary> Pandas </summary>
<p>

***[pandas](https://pandas.pydata.org/) is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.***
</p>
</details>

***Install `pandas` by :***
```python
pip install pandas
```

<details><summary> Scikit-Learn </summary>
<p>

***[Scikit-learn](https://scikit-learn.org/) (a.k.a. sklearn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support-vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.***

</p>
</details>

***Install `scikit-learn` by :***
```python
pip install -U scikit-learn
```

***In order to check your installation you can use***
```python
pip show scikit-learn  # to see which version and where scikit-learn is installed
python -c "import sklearn; sklearn.show_versions()"
```

<details><summary> Matplotlib </summary>
<p>

***[Matplotlib](https://matplotlib.org/) is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK.***

***for Matplotlib and Python see [Python Tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)***

</p>
</details>

***Install `matplotlib` by :***
```python
pip install matplotlib
```

<details><summary> tkinter </summary>
<p>

***[tkinter](https://docs.python.org/3/library/tkinter.html#module-tkinter) (“Tk interface”) is the standard Python interface to the Tcl/Tk GUI toolkit. Both Tk and tkinter are available on most Unix platforms, including macOS, as well as on Windows systems.***

***Running python -m tkinter from the command line should open a window demonstrating a simple Tk interface, letting you know that tkinter is properly installed on your system, and also showing what version of Tcl/Tk is installed.***

> ***tkinter is an in-build library module - installation is not required***

</p>
</details>

# Running the Project

## Running on Anaconda

***1. Open Anaconada Distribution and find Spyder IDE.***
***2. Load the MainCode.py and run through the Code for any errors***

> ***Note: Before Running, Make sure that both Data Set (Testing.csv and Training.csv) and MainCode.py reside under the same folder!***

***3. Lookup for an Option saying "Run in New Kernel" and run the Code***

## Running on Visual Studio / Visual Studio Code

<details><summary> Get it Done with a Click </summary>
<p>

***1. Search for the extension "Code Runner" in Visual Studio Code or you can head to [Microsoft Marketplace](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) or [GitHub](https://github.com/formulahendry/vscode-code-runner)***

***2. After instaling the Extension you can run the MainCode.py from the Run button that pops on the Top Right Corner of the Editor***

</p>
</details>

<details><summary> Terminal </summary>
<p>

***You can also Run the Code in Terminal as below :***
```
& <path-to-python-installation-directory>/python.exe <path-to-code>/MainCode.py
```

</p>
</details>