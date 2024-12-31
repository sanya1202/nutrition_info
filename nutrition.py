import uvicorn
from fastapi import FastAPI, File, UploadFile
import google.generativeai as genai
import pandas as pd
import tempfile
import json
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Configure the API key
api_key = 'AIzaSyAuuy457QvpMk9_AY3d3iOaIH-S03yhxFQ'  # Replace with your actual API key
genai.configure(api_key=api_key)

# Define the app object
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this for production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload image function
def upload_image(image_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name

    sample_file = genai.upload_file(temp_file_path)
    return sample_file

def get_nutrition_and_ingredients(sample_file):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    response = model.generate_content([
        sample_file,
        """Extract all the details regarding nutrients and ingredients shown in the given image and give me response in the format of JSON Object where Nutrient:(given nutrient) and Ingredients:(given ingredients) and only these two things and only JSON response ,no other thing. like as follows:
        {
            {
            "Nutrients": {
                "Energy": "457.7 kcal",
                "Protein": "22.20 g",
                "Total Carbohydrate": "53.3 g",
                "Total Sugars": "2.3 g",
                "Added Sugars": "0.0 g",
                "Dietary Fibre": "12.3 g",
                "Total Fat": "17.3 g",
                "Saturated Fat": "6.1 g",
                "Trans Fat": "<0.1 g",
                "Sodium": "571.3 mg"
            },
            "Ingredients": [
                "Split Green Gram (Moong Dal)",
                "Vegetable Oil (Cotton Seed, Groundnut and Rice Bran)",
                "Iodized Salt"
              ]
            }
        }"""
    ])

    response_text = response.text.strip()
    print(f"Generated nutrition and ingredients info: {response_text}")  # Debugging output

    try:
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        parsed_response = json.loads(response_text)
        return parsed_response
    except json.JSONDecodeError:
        print("Failed to parse response text as JSON.")
        return {}

def parse_response_to_dataframe(response_data):
    try:
        nutrients = response_data.get("Nutrients", {})
        ingredients = response_data.get("Ingredients", [])

        # Create a DataFrame for nutrients
        nutrients_df = pd.DataFrame([nutrients])
        nutrients_df = nutrients_df.T.reset_index()
        nutrients_df.columns = ["Nutrient", "Value"]

        # Create a DataFrame for ingredients
        ingredients_df = pd.DataFrame({"Ingredient": ingredients})

        # Combine nutrients and ingredients into one DataFrame
        combined_df = pd.concat(
            [nutrients_df, ingredients_df],
            keys=["Nutrients", "Ingredients"],
            axis=0,
            ignore_index=True,
        )

        return combined_df

    except Exception as e:
        print(f"Error in parse_response_to_dataframe: {e}")
        return pd.DataFrame()

# Define index route
@app.get('/')
def index():
    return {'message': 'Image API - Send an image to /predict'}

# Define the image prediction route
@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        sample_file = upload_image(contents)

        if not sample_file:
            return {"error": "Failed to upload the image"}
        else:
            response_data = get_nutrition_and_ingredients(sample_file)
            if response_data:
                df = parse_response_to_dataframe(response_data)
                if df.empty:
                    return {
                        "filename": file.filename,
                        "message": "No relevant product details found."
                    }
                return {
                    "filename": file.filename,
                    "message": "Product details extracted successfully.",
                    "product_details": df.to_dict(orient='records')
                }
            else:
                return {"error": "Unable to extract product details."}

    except Exception as e:
        return {"error": str(e)}

# Run the API with Uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
