# tensorflowchatbot
This chatbot was made after completing the 5th lesson of "Build Chatbots with Python" course on Codecademy. This chatbot utilizes the Keras API to create a Seq2Seq model that accepts user input as a message and forms a response based on the sample.txt file. The sample.txt file is based on a movie corpus dataset, found here:
https://github.com/Phylliida/Dialogue-Datasets

This chatbot operates on the Discord platform in the form of a Discord bot, where the '$hello' command is typed in a channel of a server the bot is added to in order to start a conversation. A screenshot of the Discord implementation is shown. 

There are several ways this project can be improved in a future implementation. For instance, the dataset used to train the model was too small and responses were evidently not human-like. This can be worked on by using a larger dataset, since the seq2seq model requires large amounts of data for it to be implemented properly. Additionally, the chatbot can only run once compiled on a local machine. For it to run continuously even when the local machine is turned off, the chatbot can be uploaded to an online cloud server such as Amazon's EC2 service, where it can run 24/7. 


