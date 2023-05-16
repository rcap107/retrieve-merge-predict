#!/bin/bash
echo "Dropping Lazo index"

curl -X GET "localhost:9250/_cat/indices/lazo*?v=true&s=index&pretty"

curl -X DELETE "localhost:9250/lazo?pretty"

curl -X GET "localhost:9250/_cat/indices/lazo*?v=true&s=index&pretty"

