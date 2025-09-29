import requests
import os

# Test the API with a sample image
def test_api():
    # Use one of the trial images
    test_image = "trial/000001.jpg"
    
    if not os.path.exists(test_image):
        print("No test image found. Please upload an image to test the API.")
        return
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:8000/predict', files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ API is working!")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"All probabilities: {result['all_probabilities']}")
        else:
            print(f"❌ API error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the API server is running on http://localhost:8000")

if __name__ == "__main__":
    test_api()
