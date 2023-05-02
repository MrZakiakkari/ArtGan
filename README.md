# Artwork Creation Using AI FYP
## How to run the project
There are three parts involved in the project which are the Artwork Creation using AI.ipynb file, the Flask application and the test cases. And I will demonstrate how to run
each part of the project.
### Jupyter Notebook(Artwork Creation using AI.ipynb)
By simply running all the cells by presseing Kernel -> Restart & Run All. When reaching the fifth cell(Train a Gan model) it will start to generate art images as well outputting
the loss functions between the generator and discriminator and showing the charts of the fid score and mode collapse detection. However, This may take time because 500 epochs is used
to train the model so it may take several hours to finish(probably 3-5 or 6 hours long).
### Flask application
To run the Flask application simply use the spyder because it supports Jupyter Notebook and by opening the app.py through spyder and then running it open the application by typing
out "http://localhost:5000/" then the application opens with a standard image and two buttons. One of the buttons is simply used to generate another image and the other button simply
saves the image being generated(ie the standard image from the application) into saved_images folder located in static folder.
### Test cases
By simply setting it to the development branch on github and then using gitbash to push the changes. By changing the directory of to where the project is stored and then
using "git status"(to check the projects status), "git add ."(To add simple changes being made which is needed to test the project), "git commit -m"(to commit the changes being made which is optional)
and lastly "git push original development" to push the changes in the github. When setting the repository to development branch and when pushing the changes a button will appear
which states"compare and pull request", by pressing that and then add some commit and then pressing "Create pull request" it will then run the test cases for the code before merging it to 
the main branch(which would probably take around 3-4 minutes). Once the test cases passed the it is safe to merge to the main branch by pressing the "Merge pull request" and 
also to confirm merging to the main branch by pressing "Confirm merge" and then the branches are succesfully merged and succesfully tested.
