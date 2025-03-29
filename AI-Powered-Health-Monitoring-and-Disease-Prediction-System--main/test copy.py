import google.generativeai as genai

genai.configure(api_key='AIzaSyBL2LtSwzQZy4VzdRzlm5YtRW-R7YIcjkM')
gemini_model = genai.GenerativeModel('gemini-1.5-pro')
chat = gemini_model.start_chat()

gemini_response = chat.send_message("you have Alcoholic hepatitis this is the disease detected throught the algorithm so you need to provide the {1) drug with dosage recommendatoin ,2) atleast 4 to 5 diet plan fr veg and non veg seperate seperately,3) write some do's and dont's ,4) if given disease is dangeoruos give doctors list near bangalore } (in html format)")
recommendatoin = gemini_response.text
recommendatoin = recommendatoin.replace("```html", "")
recommendatoin = recommendatoin.replace("```", "")
print(recommendatoin)