# Overview

This project is an exercise to train ChatGpt for a specific data model. Instructions for the step by step in preparing the data model for training is located in [The file tuning guilde](https://platform.openai.com/docs/guides/fine-tuning).  

This guide uses the OpenAI CLI tool.  

Installation: `pip install --upgrade openai`
Set API Key: `export OPENAI_API_KEY="<OPENAI_API_KEY>"`

Then prepare data.  Have a look at the *-model.jsonl file.

Run the model data through OpanAI tool: `openai tools fine_tunes.prepare_data -f <LOCAL_FILE>`
Start the fine tune job: `openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m <BASE_MODEL>`

Once fine tunes create is running or done running, the output in the terminal will look similar to below:

`
(Ctrl-C will interrupt the stream, but not cancel the fine-tune)
[2023-02-28 19:16:26] Created fine-tune: ft-83VzUOdg3QV6JRXSxf3syOoV

Stream interrupted (client disconnected).
To resume the stream, run:

  openai api fine_tunes.follow -i ft-83VzUOdg3QV6JRXSxf3syOoV
`

Where BASE_MODEL is the name of the base model you're starting from (ada, babbage, curie, or davinci). You can customize your fine-tuned model's name using the suffix parameter.

Running the above command does several things:

Uploads the file using the files API (or uses an already-uploaded file)
Creates a fine-tune job
Streams events until the job is done (this often takes minutes, but can take hours if there are many jobs in the queue or your dataset is large)
Every fine-tuning job starts from a base model, which defaults to curie. The choice of model influences both the performance of the model and the cost of running your fine-tuned model. Your model can be one of: ada, babbage, curie, or davinci. Visit our pricing page for details on fine-tune rates.

After you've started a fine-tune job, it may take some time to complete. Your job may be queued behind other jobs on our system, and training our model can take minutes or hours depending on the model and dataset size. If the event stream is interrupted for any reason, you can resume it by running: `openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID>`


When the job is done, it should display the name of the fine-tuned model.  In addition to creating a fine-tune job, you can also list existing jobs, retrieve the status of a job, or cancel a job.

# List all created fine-tunes
openai api fine_tunes.list

# Retrieve the state of a fine-tune. The resulting object includes
# job status (which can be one of pending, running, succeeded, or failed)
# and other information
openai api fine_tunes.get -i <YOUR_FINE_TUNE_JOB_ID>

# Cancel a job
openai api fine_tunes.cancel -i <YOUR_FINE_TUNE_JOB_ID>

When a job has succeeded, the fine_tuned_model field will be populated with the name of the model. You may now specify this model as a parameter to our [Completions API](https://platform.openai.com/docs/api-reference/completions/create), and make requests to it using the Playground.

You can start making requests by passing the model name as the model parameter of a completion request:

OpenAI CLI:
`openai api completions.create -m <FINE_TUNED_MODEL> -p <YOUR_PROMPT>`
