#!/bin/bash
set -e # Exit if any errors occurr
CMD=$0
read -r -d '\0' DOCUMENTATION << EOF
Used for generating documentation for the project

Usage:
  $CMD generate - Runs pydoc over all modules
  $CMD deploy - Using aws-cli, deploys documentation to an S3 bucket

Environment Variables:
  BUCKET_NAME - The name of the bucket to be deployed to (Default: current branch name)
\0
EOF

# Generates documentation for each module
function generate {
    pdoc3 --html --force covid
    pdoc3 --html --force discrete
    pdoc3 --html --force generate_discrete
    pdoc3 --html --force generate_master
    pdoc3 --html --force generate_ohe
    pdoc3 --html --force generate_predict
    pdoc3 --html --force ohe
    pdoc3 --html --force predict
}

# Deploys generated content to an AWS bucket
function deploy {
    if [ -z $BUCKET_NAME ]; then
	BUCKET_NAME=$(git branch | grep '*' | awk '{print $2}')
    fi
    BUCKET_NAME="vl-ml-doc-$(echo $BUCKET_NAME | sed 's/origin\///g')"
    BUCKET_NAME="$(echo $BUCKET_NAME | sed 's/[\/_]/-/g')"
    echo "Bucket name is $BUCKET_NAME"

    ## DO BUILD
    is_bucket_created="$(aws s3 ls | grep $BUCKET_NAME)" || true
    if [ ! -z "$is_bucket_created" ]; then
	echo "Bucket exists, removing old bucket."
	aws s3 rb s3://$BUCKET_NAME --force
    fi
    echo "Creating bucket."
    aws s3api create-bucket --acl public-read --bucket "$BUCKET_NAME" --region us-east-1
    echo "Setting bucket to be a static website."
    aws s3api put-bucket-website --bucket $BUCKET_NAME \
	--website-configuration "{\"IndexDocument\": {\"Suffix\": \"index.html\"}, \"ErrorDocument\": {\"Key\": \"index.html\"}}"

    echo "Copying html files."
    # Copies/syncs over the the compiled files making them public for the static website
    aws s3 sync html/ s3://$BUCKET_NAME --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers
    echo "http://$BUCKET_NAME.s3-website-us-east-1.amazonaws.com/"




}

ACTION=$1

case "$ACTION" in
    generate) generate;;
    deploy) deploy;;
  *)
  echo "$DOCUMENTATION"
  exit 1
esac
