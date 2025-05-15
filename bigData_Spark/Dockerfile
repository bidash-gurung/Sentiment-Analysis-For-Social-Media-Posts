# Use the latest Bitnami Spark image as the base
FROM bitnami/spark:latest

# Switch to root user to install additional packages
USER root

# Update the package list, install python3-pip, numpy, and clean up cache to reduce image size
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install numpy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Streamlit for running interactive applications
RUN pip3 install streamlit

# Switch back to non-root user for better security
USER 1001
