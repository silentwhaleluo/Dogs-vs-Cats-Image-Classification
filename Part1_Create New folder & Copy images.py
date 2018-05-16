
# coding: utf-8

# In[1]:


'''
Creating new folders and copy images
The folder structure is like this
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    test/
        test/
            001,jpg

Using the former 2/3 as training set, later 1/3 as validation set
- put the cat pictures index 0-8332 in data/train/cats
- put the cat pictures index 8333-12499 in data/validation/cats
- put the dogs pictures index 0-8332 in data/train/dogs
- put the dog pictures index 8333-12499 in data/validation/dogs
'''

path_folder='\\final_dogs&cats'
path_img = '\\dogs&cats\\train'
path_test = '\\dogs&cats\\test1\\test1'
#split rate can be change as you want such as 1/4
split_rate = 2/3

import os,shutil
print('imported')

#create new folders
def Createfolder(path_folder):
    if os.path.exists(path_folder+os.sep+'data'):
        print('folder already exsisted')
    else:
        os.makedirs(path_folder+os.sep+'data'+os.sep+'train'+os.sep+'dogs')
        os.makedirs(path_folder+os.sep+'data'+os.sep+'train'+os.sep+'cats')
        os.makedirs(path_folder+os.sep+'data'+os.sep+'validation'+os.sep+'dogs')
        os.makedirs(path_folder+os.sep+'data'+os.sep+'validation'+os.sep+'cats')    
        os.makedirs(path_folder+os.sep+'data'+os.sep+'test'+os.sep+'test')
        print('New folder created ')
    #move the images

#copy pictures
def CopyImages(path_img,path_test):
    ls = os.listdir(path_img)
    print('Training Number:',len(ls))
    for picture in ls:
        if picture.find('cat') != -1:
            if int(picture[4:-4])<int(len(ls)/2*split_rate):                
                shutil.copy(path_img+os.sep+picture,path_folder+os.sep+'data'+os.sep+'train'+os.sep+'cats'+os.sep+picture)
            else:
                shutil.copy(path_img+os.sep+picture,path_folder+os.sep+'data'+os.sep+'validation'+os.sep+'cats'+os.sep+picture)
        else: 
            if int(picture[4:-4])<int(len(ls)/2*split_rate):
                shutil.copy(path_img+os.sep+picture,path_folder+os.sep+'data'+os.sep+'train'+os.sep+'dogs'+os.sep+picture)
            else:
                shutil.copy(path_img+os.sep+picture,path_folder+os.sep+'data'+os.sep+'validation'+os.sep+'dogs'+os.sep+picture)
    print('Training images copied')
    ls_test= os.listdir(path_test)
    print('Test Number:',len(ls_test))
    for j in ls_test: 
        shutil.copy(path_test+os.sep+j,path_folder+os.sep+'data'+os.sep+'test'+os.sep+'test'+os.sep+j)
    print('Test images copied')

Createfolder(path_folder)


# In[10]:


CopyImages(path_img,path_test)
print('All pictures copied')


# In[11]:


#test the the images number in the folder
print('Number of pictures in training dogs file' ,len(os.listdir(path_folder+os.sep+'data'+os.sep+'train'+os.sep+'dogs')))
print('Number of pictures in training cats file' ,len(os.listdir(path_folder+os.sep+'data'+os.sep+'train'+os.sep+'cats')))
print('Number of pictures in validation dtss file' ,len(os.listdir(path_folder+os.sep+'data'+os.sep+'validation'+os.sep+'dogs')))
print('Number of pictures in validation cats file' ,len(os.listdir(path_folder+os.sep+'data'+os.sep+'validation'+os.sep+'cats')))
print('Number of pictures in test file' ,len(os.listdir(path_folder+os.sep+'data'+os.sep+'test'+os.sep+'test')))

