## Step 6: Set terminal chatbot client


import requests

city = input("greeting! Do you want to check the weather in London or Birmingham? ").strip().lower()


if city == "london":
    last_days = [[11, 3, 6], [11, 1, 5], [13, 0, 6], [14, 0, 6], [15, 1, 7],
                [18, 3, 10], [18, 9, 13], [17, 8, 12], [17, 9, 12], [13, 8, 10],
                [9, 5, 7], [8, 4, 6], [7, 2, 4], [8, 0, 4]]#, [8, 2, 5]]
                #[9, 0, 5], [8, 4, 6], [10, 3, 6]] #london round up
elif city == "birmingham":
    last_days = [[10, -4, 3], [10, -3, 4], [12, -3, 5], [12, 0, 6], [14, 0, 7],
                 [16, 2, 9], [16, 8, 11], [17, 7, 11], [18, 4, 11], [10, 6, 7],
                 [8, 4, 6], [7, 2, 4], [7, 1, 3], [8, 1, 5]]#, [8, 1, 4], 
                 #[8, -2, 4], [7, 4, 5], [12, -2, 5]]
# Send request to the server
response = requests.post("http://127.0.0.1:5062/predict", json={"city": city, "data": last_days})

# Getting predicted data from the server
if response.status_code == 200:
    result = response.json()
    print(f"Oracle: {result['city']} tomorrow weather is:")
    print(f"The highest temperature is {result['prediction']['tempmax']:.2f} Celsius")
    print(f"The lowest temperature is {result['prediction']['tempmin']:.2f} celsius")
    print(f"The noon temperature is {result['prediction']['temp']:.2f}")
else:
    print("Please enter the correct city name")
