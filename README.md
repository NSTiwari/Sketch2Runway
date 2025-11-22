# Fashion Sketch2Runway 
Sketch2Runway is a project that showcases Google's generative AI models **Gemini 2.0 Flash** and **Veo 3** for fashion design.
It converts your sketch to a beautiful fashion photo which you can edit and then turn it into a stunning 8-second video 
with the start-of-art video generation model Veo 3.

## Results:
<img src="https://github.com/NSTiwari/Sketch2Runway/blob/main/static/images/sketch2runway.gif"/>

The repo includes a Flask web app and a Colab notebook: 
* The **Flask web app** is perfect for anyone: whether you are an artist with creative flair or someone who can barely draw, 
try it out to see your sketch transformed to photorealistic images and then come to life with stunning runway videos.
* The **Colab notebook** contains detailed step-by-step tutorial with sample code.

## Pre-requisites:

1. Google Cloud SDK [(gcloud CLI)](https://cloud.google.com/sdk/docs/install) installed for authentication.
   
   - Go to terminal/command prompt and enter the command: `gcloud init` and choose the project ID.
     
   - Enter the following command to set a default login: `gcloud auth application-default login`.

3. Access to Veo 3 which is now available for public preview.

## Run the Web App:

1. Clone the repository on your local machine.
2. Navigate to `cd sketch2runway` directory.
3. Run `pip install -r requirements.txt` to install the packages.
4. Open `.env` file and configure your `GEMINI_API_KEY`.
5. Run `flask run` to start the server.
6. Open `localhost:5000` on your web browser to bring up the web UI. 

## Tutorials
* [YouTube tutorial](https://youtu.be/9Bo5sEVoVQk)
* [Blog post tutorial](https://margaretmz.medium.com/fashion-sketch2runway-with-gemini2flash-veo3-ced1e2776fea)
