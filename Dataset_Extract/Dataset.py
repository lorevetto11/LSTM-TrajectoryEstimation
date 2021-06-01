import numpy as np
import glob
import json
import os
from Interpolation import interpolation

class Dataset:

    def __init__(self):
        self.interpolation = interpolation.Interpolation()
        self.dataset_files = []
        self.dataset_files2 = []
        self.dataset_files3 = []
        self.dataset_files_tra = []
        self.mod_dataset = []

        self.x1, self.x2, self.y1, self.y2, self.x_goal, self.y_goal = [], [], [], [], [], []
        self.x1_test, self.x2_test, self.y1_test, self.y2_test = [], [], [], []

        self.X, self.Y = [], []

        self.myNewX, self.myNewY = [], []

        self.tmp_x, self.tmp_y = [], []

        self.test, self.test2, self.test3, self.test4 = [], [], [], []

    def extract_F1_dataset_IdealLine(self):

        for document in glob.glob('F1_Dataset/*_track.csv'):
            self.dataset_files.append(document)

        for document in glob.glob('F1_Dataset/*_trajectory.csv'):
            self.dataset_files_tra.append(document)

        for file, file1 in zip(self.dataset_files, self.dataset_files_tra):
            csv_data_temp = np.loadtxt(file, comments='#', delimiter=',')
            csv_data_temp2 = np.loadtxt(file1, comments='#', delimiter=',')

            mod = int(len(csv_data_temp) / 10)

            interp_track_left = self.interpolation.interpolate_polyline(csv_data_temp[:, 0], csv_data_temp[:, 1], int(len(csv_data_temp2)))
            interp_track_right = self.interpolation.interpolate_polyline(csv_data_temp[:, 2], csv_data_temp[:, 3], int(len(csv_data_temp2)))


            self.x1.append(interp_track_left[:, 0])
            self.y1.append(interp_track_left[:, 1])
            self.x2.append(interp_track_right[:, 0])
            self.y2.append(interp_track_right[:, 1])

            self.x_goal.append(csv_data_temp2[:, 0])
            self.y_goal.append(csv_data_temp2[:, 1])

            self.data_augmentation(interp_track_left[:, 0], interp_track_left[:, 1], interp_track_right[:, 0], 
                interp_track_right[:, 1], csv_data_temp2[:, 0], csv_data_temp2[:, 1])

    def extract_F1_dataset(self):

        for document in glob.glob('F1_Dataset/*_track.csv'):
            self.dataset_files.append(document)

        for document in glob.glob('F1_Dataset/*_trajectory.csv'):
            self.dataset_files_tra.append(document)

        for file, file1 in zip(self.dataset_files, self.dataset_files_tra):
            csv_data_temp = np.loadtxt(file, comments='#', delimiter=',')
            csv_data_temp2 = np.loadtxt(file1, comments='#', delimiter=',')

            self.x1.append(csv_data_temp[:, 0])
            self.y1.append(csv_data_temp[:, 1])
            self.x2.append(csv_data_temp[:, 2])
            self.y2.append(csv_data_temp[:, 3])

            #self.x_goal.append(csv_data_temp2[:, 0])
            #self.y_goal.append(csv_data_temp2[:, 1])

            #self.data_augmentation(interp_track_left[:, 0], interp_track_left[:, 1], interp_track_right[:, 0], 
                #interp_track_right[:, 1], csv_data_temp2[:, 0], csv_data_temp2[:, 1])

    def extract_MOD_dataset(self):

        for document2 in glob.glob('Mod_Dataset/*.json'):
            self.mod_dataset.append(document2)

        for l in self.mod_dataset:
            print(l)
            with open(l) as json_file:
                data = json.load(json_file)
                inner = data["inner"]
                outer = data["outer"]
                for i in inner:
                    self.test.append(i[0])
                self.test = np.array(self.test)
                self.x1.append(self.test)
                self.test = []
            
                for i in inner:
                    self.test2.append(i[1])
                self.test2 = np.array(self.test2)
                self.y1.append(self.test2)
                self.test2 = []

                for i in outer:
                    self.test3.append(i[0])
                self.test3 = np.array(self.test3)
                self.x2.append(self.test3)
                self.test3 = []

                for i in outer:
                    self.test4.append(i[1])
                self.test4 = np.array(self.test4)
                self.y2.append(self.test4)
                self.test4 = []
        
            
    def extract_AC_dataset(self):
        for document2 in glob.glob('AC_Dataset/*_side_l.csv'):
            self.dataset_files2.append(document2)

        for document2 in glob.glob('AC_Dataset/*_side_r.csv'):
            self.dataset_files3.append(document2)

        for i in range(0, len(self.dataset_files2)):

                csv_data_temp1 = np.loadtxt(self.dataset_files2[i], comments='#', delimiter=',')

                csv_data_temp2 = np.loadtxt(self.dataset_files3[i], comments='#', delimiter=',')
                
                if(len(csv_data_temp1[:, 0]) >= len(csv_data_temp2[:, 0])):

                    mod = int(len(csv_data_temp1[:, 0])/10)

                    interp_path_left = self.interpolation.interpolate_polyline(csv_data_temp1[:, 0], csv_data_temp1[:, 2], mod*10)
                    interp_path_right = self.interpolation.interpolate_polyline(csv_data_temp2[:, 0], csv_data_temp2[:, 2], mod*10)

                    self.x1.append(interp_path_left[:, 0])
                    self.y1.append(interp_path_left[:, 1])

                    self.x2.append(interp_path_right[:, 0])
                    self.y2.append(interp_path_right[:, 1])

                    self.data_augmentation(interp_path_left[:, 0], interp_path_left[:, 1], interp_path_right[:, 0], 
                        interp_path_right[:, 1], csv_data_temp2[:, 0], csv_data_temp2[:, 1])

                else:
                    mod = int(len(csv_data_temp2[:, 0])/10)

                    interp_path_left = self.interpolation.interpolate_polyline(csv_data_temp1[:, 0], csv_data_temp1[:, 2], mod*10)
                    interp_path_right = self.interpolation.interpolate_polyline(csv_data_temp2[:, 0], csv_data_temp2[:, 2], mod*10)

                    self.x1.append(interp_path_left[:, 0])
                    self.y1.append(interp_path_left[:, 1])

                    self.x2.append(interp_path_right[:, 0])
                    self.y2.append(interp_path_right[:, 1])

                    self.data_augmentation(interp_path_left[:, 0], interp_path_left[:, 1], interp_path_right[:, 0], 
                        interp_path_right[:, 1], csv_data_temp2[:, 0], csv_data_temp2[:, 1])
         
    def extract_center_trajectory(self):
        for i in range(0, len(self.x1)):
            self.x_goal.append((self.x1[i] + self.x2[i]) / 2)
            self.y_goal.append((self.y1[i] + self.y2[i]) / 2)

    def reshape_one_array(self):
        for i in range(0, len(self.x1)):
            self.X.append(np.column_stack((self.x1[i], self.y1[i], self.x2[i], self.y2[i])))
            self.myNewX.append(self.X[i].reshape(len(self.X[i]),1,4))

            self.Y.append(np.column_stack((self.x_goal[i], self.y_goal[i])))
            self.myNewY.append(self.Y[i].reshape(len(self.Y[i]), 2))

    def reshape_multi_array(self):
        for i in range(0, len(self.x1)):
            self.X.append(np.column_stack((self.x1[i], self.y1[i], self.x2[i], self.y2[i])))
            self.myNewX.append(self.X[i].reshape(int(len(self.X[i])/10),10,4))

            self.Y.append(np.column_stack((self.x_goal[i], self.y_goal[i])))
            self.myNewY.append(self.Y[i].reshape(int(len(self.Y[i])/10), 20))

    def get_dataset(self):
        return self.myNewX, self.myNewY

    def circuit_flip_vertical(self, x1, y1, x2, y2):

        '''
        Flip Vertical
        '''
        self.x1.append(x1 * -1)
        self.y1.append(y1)
        self.x2.append(x2* -1)
        self.y2.append(y2)

    def circuit_flip_horizontal(self, x1, y1, x2, y2):

        '''
        Flip Horizontal
        '''
        self.x1.append(x1 )
        self.y1.append(y1* -1)
        self.x2.append(x2)
        self.y2.append(y2* -1)

    def circuit_flip_horizontal_vertical(self, x1, y1, x2, y2):

        '''
        Flip Horizontal and Vertical
        '''
        self.x1.append(x1* -1)
        self.y1.append(y1* -1)
        self.x2.append(x2* -1)
        self.y2.append(y2* -1)
  
    def data_augmentation(self, x1, y1, x2, y2, xg, yg):


        tmp_x1 = np.negative(x1)
        tmp_x2 = np.negative(x2)
        tmp_y1 = np.negative(y1)
        tmp_y2 = np.negative(y2)

        tmp_xg = np.negative(xg)
        tmp_yg = np.negative(yg)

        '''
        Flip V e O
        '''
        self.x1.append(tmp_x1)
        self.y1.append(tmp_y1)
        self.x2.append(tmp_x2)
        self.y2.append(tmp_y2)

        self.x_goal.append(tmp_xg)
        self.y_goal.append(tmp_yg)

        '''
        Flip V 
        '''
        self.x1.append(tmp_x1)
        self.y1.append(y1)
        self.x2.append(tmp_x2)
        self.y2.append(y2)

        self.x_goal.append(tmp_xg)
        self.y_goal.append(yg)

        '''
        Flip O
        '''
        self.x1.append(x1)
        self.y1.append(tmp_y1)
        self.x2.append(x2)
        self.y2.append(tmp_y2)

        self.x_goal.append(xg)
        self.y_goal.append(tmp_yg)