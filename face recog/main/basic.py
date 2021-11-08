import cv2
import numpy
import face_recognition

imgElon = face_recognition.load_image_file('images/elon.jpeg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images/elon_test.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


# to convert image to RGB


faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]),
              (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# print(faceLoc)  gives face cordinates
# print(encodeElon) gives all 128 parameter values

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]),
              (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# print(faceLocTest) gives face cordinates
# print(encodeTest) gives all 128 parameter values

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)

# compares the 2 faces

cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (
    50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255, 2))

cv2.imshow('elon', imgElon)
cv2.imshow('elon_test', imgTest)


cv2.waitKey(0)
