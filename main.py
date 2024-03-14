
# File: main
# Author: Taylor Kramer
# Date: 03/02/24

'''Prompt: Creating and testing machine learning models to predict the direction of stocks.
   Answering the question: Will a certain stock go up or down tomorrow? How well can 
   various machine learning models make these predictions.''' 

from interact import Interact

def main():
   
   '''Creates instance of the interact class which is the class responsible for running and
   interacting with all other classes within the program. Begins running the program by
   calling the function that runs the user interface.'''
   
   myjourney = Interact()
   myjourney.user_interface()
   
   return

if __name__ == "__main__":
   main()