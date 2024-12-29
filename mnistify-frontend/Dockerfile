# Use the official Node.js image as a parent image
FROM node:18-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the Svelte app's source code
COPY . .

# Expose the port the app will run on inside the container
EXPOSE 8080

# Start the Svelte app in development mode
CMD ["npm", "run", "dev", "--", "--port", "5000", "--host"]

