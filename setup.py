import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()
    
REQUIREMENTS = ["numpy", "opencv-python", "alfred-py"]

setuptools.setup(
     name='smpl2',  
     version='0.0.1',
     author="Lucas Jin",
     author_email="11@qq.com",
     install_requires=REQUIREMENTS,
     description="SMPL2: Optimized common used SMPL operation.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/jinfagang/SMPL2",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 2",
         "Programming Language :: Python :: 3",
         "License :: Other/Proprietary License",
         "Operating System :: OS Independent",
     ],
     

 )
