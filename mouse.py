import cv2
import mediapipe as mp
import pyautogui

""""
This Program implements mediapipe's pre-trained solutions and pyautogui's mouse functionality 
for using fingers as mouse.
INDEX FINGER + THUMB = LEFT CLICK
MIDDLE FINGER + THUMB = RIGHT CLICK

Thanks for using, any contributions are appreciated.



"""




cam = cv2.VideoCapture(0)
fin_dect = mp.solutions.hands.Hands() #type: ignore
draw_hand = mp.solutions.drawing_utils #type: ignore
screen_w, screen_h = pyautogui.size()
ind3x_y = 0
thumb_y = 0




while True:
    _,    fram3 = cam.read()
#----------------------------------------------------------------    
    fram3 = cv2.flip(fram3, 1)
#----------------------------------------------------------------     
    fram3_h, fram3_w, _ = fram3.shape
#----------------------------------------------------------------     
    clr_fram3 = cv2.cvtColor(fram3, cv2.COLOR_BGR2RGB)
#----------------------------------------------------------------      
    oup = fin_dect.process(clr_fram3)
#----------------------------------------------------------------      
    hands = oup.multi_hand_landmarks
#----------------------------------------------------------------  


    
    if hands: # hand detection logic 
        #----------------------------------------------------------------  
        for hand in hands:
            #----------------------------------------------------------------  
            draw_hand.draw_landmarks(fram3, hand)
            #----------------------------------------------------------------  
            lndmrks = hand.landmark
            #----------------------------------------------------------------  
            
            for id, landmark in enumerate(lndmrks):
                #----------------------------------------------------------------  
                x = int(landmark.x*fram3_w)
                y = int(landmark.y*fram3_h)
                #----------------------------------------------------------------  
                
                if id == 8: #index finger logic
                    #----------------------------------------------------------------  
                    cv2.circle(img=fram3, center=(x, y), radius=10, color=(70, 100, 200))
                    #----------------------------------------------------------------  
                    ind3x_x = screen_w/fram3_w*x
                    ind3x_y = screen_h/fram3_h*y
                    #----------------------------------------------------------------  
                    pyautogui.moveTo(ind3x_x, ind3x_y) #mouse pointer logic
                  
#--------------------------------------------------------------------------------------------------------------------------------

                if id == 4: #index finger
                    #----------------------------------------------------------------  
                    cv2.circle(img=fram3, center=(x, y), radius=10, color=(0, 255, 255))
                    #----------------------------------------------------------------  
                    thumb_x = screen_w/fram3_w*x
                    thumb_y = screen_h/fram3_h*y
                    #----------------------------------------------------------------  
                    print('ABSOLUTE DISTANCE BETWEEN INDEX AND THUMB', abs(ind3x_y - thumb_y))
                    
                    
                    if abs(ind3x_y - thumb_y) < 20: #left click's logic
                        #----------------------------------------------------------------  
                        print("LEFT-CLICK ACTION INITIATED")
                        #----------------------------------------------------------------  
                        pyautogui.leftClick()
                        pyautogui.sleep(1)
                    #----------------------------------------------------------------  
                    
                if id == 12: #middle finger's logic
                    cv2.circle(img=fram3, center=(x, y), radius=10, color=(100, 83, 270))
                    #----------------------------------------------------------------  
                    middle_x = screen_w/fram3_w*x
                    middle_y = screen_h/fram3_h*y   
                    #----------------------------------------------------------------  
                    print("ABSOLUTE DISTANCE BETWEEN MIDDLE FINGER AND THUMB", abs(middle_y - thumb_y))
                    
                    
                    if abs(middle_y - thumb_y) < 20: #right click's logic
                        #----------------------------------------------------------------  
                        print("RIGHT-CLICK ACTION INITIATED")
                        #----------------------------------------------------------------  
                        pyautogui.rightClick()
                        pyautogui.sleep(1)
                        #----------------------------------------------------------------  
                    
        

    cv2.imshow('Erouse', fram3) 
    cv2.waitKey(1)
    
    


#