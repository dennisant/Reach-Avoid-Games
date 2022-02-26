# Instruction to use Git

Instruction to use Git:

## After you have finished editing:
Run the following command to stage all your changes

~~~
git add .
~~~

## After you have staged your changes
Run the following command to commit, with a message
~~~
git commit -m "your message here"
~~~

## After you have commited
Run the following command to send all changes to the cloud
~~~
git push
~~~

## To get changes made by others
Run the following command to get the most recent version of the code on the cloud
~~~
git pull
~~~
**IMPORTANT**: When you make some changes on your computer and then git push, if other people have done that before you, you are behind the main commits and git push will not work. You will have to pull first, fix all the conflicts (the places that both you and your friends make changes), then add all the files again and push again using:
~~~
git add .
git commit -m "some message"
git push
~~~

## To create new branch
~~~
git checkout -b <branch_name>
~~~

## To switch between branches
~~~
git checkout <branch_name>
~~~
**IMPORTANT**: There is no "-b" flag because you are switching between branches that are available. To check for available branches, run:
~~~
git branch
~~~

## To check for general information of your current work
~~~
git status
~~~

## To go to a commit in the past and start working from there
Run the following command to see all your previous commits
~~~
git log
~~~

Once you have found your commit to go back to, copy the hash key of that commit, and then press "q" to go out of git log. Then type:
~~~
git checkoout -b <branch_name> <hash_key_of_commit>
~~~
Then you will be on another branch, with the chosen commit at your beginning.