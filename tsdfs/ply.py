import numpy as np
import os

class Ply(object):
    """
    Class to represent a ply in memory, read plys, and write plys.
    """
    
    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point. Defaults to None.
        """
        super().__init__()
       
        if ply_path == None:
            self.triangles = triangles
            self.points = points
            self.normals= normals
            self.colors = colors
        else:
            points = []
            normals =[]
            colors = []

            self.ply_path = ply_path
            List=[]
            
            with open (ply_path, 'rt') as file:  
                for line in file:
                    Num_list = []              
                    List.append(line) #Store all of contents in here  
                    bool= all(line.isdigit())
                    if bool:
                        for char in line:
                            if char!= ' ':
                                Num_list.append(int(char)) #Appends integers into num_list for each line
                        points.append(Num_list[:2]) #Num_list has length 3 here
                        if len(Num_list == 6) and 'property float nx' in List: #if it has normals, append
                            normals.append(Num_list[3:5])
                        if len(Num_list == 6) and 'property float nx' not in List: #if it has no normals but has colors
                            normals.append(Num_list[3:5])
                        if len(Num_list == 9): #this implies it has both normals and colors
                            colors.append(Num_list[6:8])
                            
            # In case len(Num_list == 6 and it isnt normals, we need to see if "property float nx" exists
                file.close()
                        

        if normals is not None and points.shape[0] != normals.shape[0]:
              raise Exception("Sorry, there isn't an equal number of points and normals")
        if colors is None and colors.shape[0] != normals.shape[0]:
            raise Exception("Sorry, there isn't an equal number of colors and normals")


    def write(self, ply_path):
        """
        Write mesh, point cloud, or oriented point cloud to ply file.
        Args:
            ply_path (str): Output ply path.
        """
        with open(ply_path, 'w') as ply:
            ply.write('ply \nformat ascii 1.0 \nelement vertex ' + str(self.points.shape[0]))
            ply.write('\nproperty float x \nproperty float y \nproperty float z\n') #To add points

            if self.normals is not None:
                ply.write('property float nx \nproperty float ny \nproperty float nz\n') #Add normals if they exist

            if self.colors is not None:
                ply.write('property uchar red \nproperty uchar green \nproperty uchar blue\n') #To add points
                
            if self.triangles is not None:
                ply.write("element face " + str((self.triangles.shape[0])) + '\nproperty list uchar int vertex_index \nend_header\n')
                #ply.write('\n ' + str(self.points) + str(self.normals) + str(self.colors))
                #ply.write('\n ' + str(self.triangles) + str(self.triangles) + str(self.triangles)) #Face list
                #print(self.triangles)
                for i in range(0,self.points.shape[0]):
                    x = self.points[i][0]
                    y = self.points[i][1]
                    z = self.points[i][2]
                    x_n = self.normals[i][0]
                    y_n = self.normals[i][1]
                    z_n = self.normals[i][2]
                    R = self.colors[i][0]
                    G = self.colors[i][1]
                    B = self.colors[i][2]
                    ply.write(str(x) +' ' + str(y) + ' ' + str(z) + ' ' + str(x_n) +' ' + str(y_n) + ' ' + str(z_n) + ' ' +  str(R) +' ' + str(G) + ' ' + str(B) +'\n')
                    
                for i in range(0,self.triangles.shape[0]):
                    v1 = self.triangles[i][0]
                    v2 = self.triangles[i][1]
                    v3 = self.triangles[i][2]
                    ply.write(str(3) +' ' + str(v1) + ' ' + str(v2) + ' ' + str(v3))
                    ply.write('\n')
            else:           
                ply.write('end_header\n')
                #ply_array =[]
                for i in range(0,self.points.shape[0]):
                    x = self.points[i][0]
                    y = self.points[i][1]
                    z = self.points[i][2]
                    x_n = self.normals[i][0]
                    y_n = self.normals[i][1]
                    z_n = self.normals[i][2]
                    R = self.colors[i][0]
                    G = self.colors[i][1]
                    B = self.colors[i][2]
                    ply.write(str(x) +' ' + str(y) + ' ' + str(z) + ' ' + str(x_n) +' ' + str(y_n) + ' ' + str(z_n) + ' ' +  str(R) +' ' + str(G) + ' ' + str(B))
                 
                    ply.write('\n')

                    
            ply.close()


    def read(self, ply_path):
        """
        Read a ply into memory.
        Args:
            ply_path (str): ply to read in.
        """
        
        if ply_path is None:
            raise Exception("Input for ply_path is none- inappropriate input")
        else:
            with open(ply_path) as ply:
                ply.read()
                

     
