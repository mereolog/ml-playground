FROM denoland/deno:2.1.9



RUN apt-get update && apt-get install -y wget

WORKDIR /deno

COPY . .

RUN deno task install:deno

CMD deno task install:theme-change && deno task build:css
