import cv2
from bar_chart import bar_draw
from detector import Detector
from color_recognition_api import color_histogram_feature_extraction, knn_classifier

class FrameAnalyzer:
    def __init__(self, cap=None):
        self.detector = Detector()
        self.color_counts = {"red": 0, "yellow": 0, "green": 0, "orange": 0, "white": 0, "black": 0, "blue": 0}
        self.detected = 0
        self.line_offset = 6   
        
        if cap is not None: 
            self.pos_line = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.763) #Posição da linha de contagem
        else:
            self.pos_line = 300

        print("analyzer initialized")

    def reset(self):
        self.color_counts = {"red": 0, "yellow": 0, "green": 0, "orange": 0, "white": 0, "black": 0, "blue": 0}
        self.detected = 0     

    def analyze(self, frame):
        frame_ = frame.copy()
        coordinates = self.detector.detect(frame)

        if coordinates is None:
            return 

        #print(self.color_counts)

        for c1, c2 in coordinates:
            roi = frame_[c1[1]:c2[1], c1[0]:c2[0], :]

            color_histogram_feature_extraction.color_histogram_of_test_image(roi)
            color = knn_classifier.main('training.data', 'test.data')

            y = (c1[1] + c2[1]) / 2

            if y<(self.pos_line+self.line_offset) and y>(self.pos_line-self.line_offset):
                self.color_counts[color] +=1
                self.detected += 1
                cv2.line(frame, (0, self.pos_line ), (frame.shape[1], self.pos_line), (230, 126, 34), 3) 

            # rendering
            cv2.rectangle(frame, c1, c2, (0,0,0), 2)
            t_size = cv2.getTextSize(color, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(frame, c1, c2,(0,0,0),-1)
            cv2.putText(frame, color, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

    def plot_results(self, path='output/result.png'):
        colors = []
        colors.append(self.color_counts["red"])
        colors.append(self.color_counts["yellow"])
        colors.append(self.color_counts["green"])
        colors.append(self.color_counts["orange"])
        colors.append(self.color_counts["white"])
        colors.append(self.color_counts["black"])
        colors.append(self.color_counts["blue"])
        bar_draw(colors, path)

if __name__ ==  "__main__":
    print("gobrrrr")
    cap = cv2.VideoCapture("video.mp4")
    fa = FrameAnalyzer(cap)
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        fa.analyze(frame)

        if not ret:
            break

        cv2.imshow("frame", frame)
        cv2.waitKey(1)

    fa.plot_results()
    print("done")
