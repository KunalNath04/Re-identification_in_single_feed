from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Crop the player patch
        x1, y1, x2, y2 = map(int, bbox)
        # clamp to frame
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        image = frame[y1:y2, x1:x2]
        # if nothing left, return a default (e.g. red) to avoid empty KMeans
        if image.size == 0:
            return np.array([0, 0, 255])

        # take top half
        th = image.shape[0] // 2
        top_half_image = image[:th, :]
        if top_half_image.size == 0:
            return np.array([0, 0, 255])

        # reshape & cluster
        image_2d = top_half_image.reshape(-1, 3)
        kmeans = self.get_clustering_model(image_2d)  # note: pass image_2d directly
        labels = kmeans.labels_.reshape(top_half_image.shape[:2])

        # find player cluster
        corners = [labels[0,0], labels[0,-1], labels[-1,0], labels[-1,-1]]
        non_player = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player

        return kmeans.cluster_centers_[player_cluster]



    def assign_team_color(self,frame, player_detections):
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id ==91:
            team_id=1

        self.player_team_dict[player_id] = team_id

        return team_id
