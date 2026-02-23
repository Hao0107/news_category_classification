from setuptools import setup, find_packages

setup(
    name="news_category_classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "pyvi",      
        "beautifulsoup4",
        "requests",
    ],
    author="Anh Hao",
    description="Dự án phân loại thể loại tin tức sử dụng các kỹ thuật NLP và machine learning.",
)