import os
import openai

openai.organization = "org-PwafGfC5oVQgjaAzYFmgp1ep"
openai.api_key =  "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"


def main():
    modelList = openai.Model.list()
    print("Training in Session")
    print("Model List " + str(modelList))
    # upload file containing data used to train Gpt
    

if __name__ == '__main__':
    main()
