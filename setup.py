from setuptools import find_packages, setup

setup(
    name="carapace",
    version="0.1.0",
    description="PR and issue triage orchestration with similarity clustering and canonical ranking",
    packages=find_packages(include=["carapace", "carapace.*"]),
    include_package_data=True,
)
