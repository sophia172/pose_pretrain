from setuptools import find_packages, setup
HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> list:
    """
    This function will return the list of requirements
    """
    try:
        requirements = []
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.replace("\n", "") for req in requirements]
            if HYPEN_E_DOT in requirements:
                requirements.remove(HYPEN_E_DOT)
        return requirements
    except FileNotFoundError:
        return []

setup(
    name="pose_pretrain",
    version="V0.0.1",
    author="Ying Liu",
    author_email="sophia.j.liu@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["pretrain=src.scripts.train:cli"],
    },
    description="End to End Machine Learning Project template",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    license="MIT",
)
