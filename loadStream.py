import cv2
import time
import threading

class loadStream:
    def __init__(self, sources='0'):
        n = 1
        self.imgs = [None] * n
        for i, s in enumerate(sources):
            url = eval(s) if s.isnumeric() else s
            cap = cv2.VideoCapture(url)
            assert cap.isOpened(), f'Failed to open {s}'

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = threading.Thread(target=self.update, args=([i, cap]), daemon=True)

            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()

    def update(self, index, cap):
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 1:
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  
    
    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'): 
            cv2.destroyAllWindows()
            raise StopIteration

        return img0

    def __len__(self):
        return 0  