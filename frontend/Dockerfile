# ==============================================================================
# Stage 1: Build Static Vite Assets
# ==============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency manifests
COPY package.json package-lock.json* ./

# Install dependencies using clean install
RUN npm ci || npm install

# Copy application sources
COPY . .

# Build production bundle
RUN npm run build

# ==============================================================================
# Stage 2: Serve Static Bundle with Ultra-lightweight Nginx Alpine
# ==============================================================================
FROM nginx:alpine AS runtime

# Remove default nginx boilerplate
RUN rm -rf /etc/nginx/conf.d/* /usr/share/nginx/html/*

# Copy custom reverse-proxy and SPA configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled static assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
