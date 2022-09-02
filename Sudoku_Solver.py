from datetime import datetime
import cv2 as cv
import streamlit as st
import numpy as np
from utils import displayNumbers, drawGrid, getPredection, intializePredectionModel, preProcess, splitBoxes, stackImages
from utils import reorder
from utils import biggestContour
import numpy as np
from sudoku import Sudoku

########################################################################
heightImg = 450
widthImg = 450
model = intializePredectionModel()  # LOAD THE CNN MODEL
########################################################################

if __name__== '__main__':

    st.title('Sudoku Solver')

    uploaded_file = st.file_uploader("Choose a sudoku image")
    container = st.container()

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()

        container.image(bytes_data)
        container.text('Unsolved Sudoku')
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)

        img = cv.imdecode(file_bytes, 1)
        original_shape = img.shape
        img = cv.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
        imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
        imgThreshold = preProcess(img)

        imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
        imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
        contours, hierarchy = cv.findContours(imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
        cv.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS

        biggest, maxArea = biggestContour(contours)
        print(biggest)
        if biggest.size != 0:
            biggest = reorder(biggest)
            # print(biggest)
            cv.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
            pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            matrix = cv.getPerspectiveTransform(pts1, pts2) # GER
            imgWarpColored = cv.warpPerspective(img, matrix, (widthImg, heightImg))
            imgDetectedDigits = imgBlank.copy()
            imgWarpColored = cv.cvtColor(imgWarpColored,cv.COLOR_BGR2GRAY)

            #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
            imgSolvedDigits = imgBlank.copy()
            boxes = splitBoxes(imgWarpColored)
            print(len(boxes))
            # cv.imshow("Sample",boxes[65])
            numbers = getPredection(boxes, model)
            print(numbers)
            # print(numbers)
            imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 0))
            numbers = np.asarray(numbers)
            posArray = np.where(numbers > 0, 0, 1)
            # print(posArray)


            #### 5. FIND SOLUTION OF THE BOARD
            try:
                board = np.array_split(numbers,9)
                # print(board)
                puzzle = Sudoku(3, 3, board=board)
                # print(puzzle)
                solution  = puzzle.solve(raising=True)
                board = solution.board
                flatList = []
                for sublist in board:
                    for item in sublist:
                        flatList.append(item)
                solvedNumbers =flatList*posArray
                imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

                # #### 6. OVERLAY SOLUTION
                pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
                pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
                matrix = cv.getPerspectiveTransform(pts1, pts2)  # GER
                imgInvWarpColored = img.copy()
                imgInvWarpColored = cv.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
                inv_perspective = cv.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
                imgDetectedDigits = drawGrid(imgDetectedDigits)
                imgSolvedDigits = drawGrid(imgSolvedDigits)

                imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                            [imgDetectedDigits, imgSolvedDigits,imgInvWarpColored,inv_perspective])
                container.success(body='Here is your solved Sudoku TADAA!!!', icon="✅")
                stackedImage = stackImages(imageArray, 1)
                container.image(stackedImage)
                inv_perspective = cv.resize(inv_perspective,(original_shape[:2]))
                container.image(inv_perspective)
                st.download_button(label="Download image",data=cv.imencode('.jpg',inv_perspective)[1].tobytes(),file_name="output"+str(datetime.now())+".png",mime="image/png")
                # cv.imshow('Stacked Images', stackedImage)
            except:
                
                container.warning(icon="⚠️",body="Either Image is not high resoluted or sudoku is unsolvable please try with a better image if former is case.")
                container.image(imgDetectedDigits)
        else:
            print("No Sudoku Found")

        cv.waitKey(0)