*** Disclaimer ***

I don't watch nor follow NFL or american football, so if there are any mistakes in the implementation, it was due to ignorance to the topic.

To run the program, you simply have to load "cleaned_main.py" with the other relevant input files to your local google storage, because by then, I would have already deleted my files to keep my credits.

Don't forget to change the names of the gs route within the .py files that you will be using so that an error would not pop up.

I don't recommend using the local environment because the program kept crashing in my new Macbook Pro m2 chip. So, if you'd like the test through the code, please do so in your Google Cluster environment. E2-standard4 for master and workers' node is enough and should run hopefully below an hour.

If you'd like to test in your local environment first, these are the files that would run without trouble: [Preliminary Research Questions.py](Preliminary%20Research%20Questions.py) and [Preprocessing_cleaneddata.py](Preprocessing_cleaneddata.py).
With these files, you'll start encountering memory problems: [Preprocessing_TrainTestBalancedData.py](Preprocessing_TrainTestBalancedData.py) and [main.py](main.py).
This file was used as a reference to make the code of the performance metrics, inspired by my Assignment 5 homework: [evaluation_depreciated.py](evaluation_depreciated.py)
And this last file was the complete working .py file that I used to make the code, here you can see all the attempts that I made along with my frustration in coding in PySpark: [main_cluster.py](main_cluster.py).
If you'd just like to see a clean final .py file, please look at: [cleaned_main.py](cleaned_main.py)

Also in the files are some of the saved cleaned datasets, train and test dataset, and treated train and test datasets stored in the form of a folder since the file is too big and the program had to break them down to chunks. I don't recommend looking at them because it will not be up-to-date. I used them to refer to their schema to double check if I'm doing it correctly.
I've also downloaded the eventLog for my final deployment of my program in the Google Cluster environment.

According to the requirements of the project, there are also 3 additional files. The .docx file contains the report for the project, the .pptx file is the slides for the presentation, and lastly, this README.md file to explain the files within this folder along with the input files that I have downloaded from: https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/overview. 
For comparison to real-life data, please feel free to explore: https://www.nfl.com/scores/2022/REG1.

Lastly, here is the breakdown of the input files that will be used by the program.
Starting with the tracking weeks from 1 to 9 csv files: these files contain full information of the plays of all the games from each week of week 1 to week 9 2022 NFL Playoffs.
tackles.csv file contains information of all the tackles, fumbles, assists, and missed tackles that occurred throughout the whole first 9 weeks of the 2022 NFL Playoffs.
players.csv file contains key information of all the players playing in the 2022 NFL Playoffs.
games.csv file contains information about the games occurring in the first 9 weeks of the 2022 NFL Playoffs, these information include the name of the clubs playing along with their final score after week 9 of the season Playoffs.
plays.csv file contains information of all the plays that happen within the first 9 weeks of the 2022 NFL Playoffs.

If there are any further questions regarding this project please feel free to contact me, Natasya Liew U15913137, through my email nliew@bu.edu.

Special thanks to Nathan Horak, Erhan Aslan, and Professor Dimitri Trajanov for guiding me through this traumatic language.