from deepface import DeepFace

# Compare two faces
result = DeepFace.verify("img1.jpg", "img1.jpg", model_name="ArcFace", detector_backend="retinaface")

# Print result
if result["verified"]:
    print("Same person ✅")
else:
    print("Different people ❌")

# print(result)

