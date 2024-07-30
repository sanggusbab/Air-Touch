gaze function Ver 1 

 The vector sum was calculated in the same way for both face and eye gaze. 

Function name: gaze
input: frame, distance
output: gaze_point_x, gaze_point_y

How it works 

Gaze at the center dot and press 'c' to set that point as the initial value. 
Based on the initial value, it shows the coordinates moved by the line of sight.
The large dot is the average position of the previous 10 coordinates (NUM_LIST), and the small dot is the most recent one. There were a lot of them, so I decided on 10, but I think I need to reduce them depending on the calculation speed. 

Face: Measure the orthographic angle using the 2D and 3D coordinates of the face -> Find the coordinates and angle of the face. 
Pupil: Compare the center of the pupil and the center of the black ruler to find the coordinates of which direction you are looking and multiply by an appropriate constant. (EYE_DEGREE_PARAMETER)  The eyes move too little, making it difficult to do so in the same way as the face. 

current situation: 
Currently, left and right are still recognized well, but it seems that top and bottom (especially bottom) are not recognized well. 
Currently, the top and bottom are weighted, but it would be nice if there was a better way.






before start that file,
here are things you need to check before.

1. python version 3.11.9
2. download './haarcascade_frontalface_default.xml' and ''./haarcascade_eye.xml' at the following link
   https://github.com/anaustinbeing/haar-cascade-files/tree/master 

