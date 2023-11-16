import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

REQUIREMENTS = ["numpy", "opencv-python", "alfred-py", "onnxruntime", "rich"]

setuptools.setup(
    name="nosmpl",
    version="0.1.4",
    author="Lucas Jin",
    author_email="11@qq.com",
    install_requires=REQUIREMENTS,
    description="NoSMPL: Optimized common used SMPL operation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jinfagang/nosmpl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
)
