AWS_REGION = us-east-1
STAGING_DIR = stage
ZIP = zip

lambda: evoldeeds-crud.lambda

%.lambda: %.zip
	aws lambda update-function-code --region $(AWS_REGION) --publish --zip-file fileb://$< --function-name $* >$@
	rm -rf $*.zip index.mjs $(STAGING_DIR)

evoldeeds-crud.zip: evoldeeds-crud.js
	(test -e $(STAGING_DIR) && rm -rf $(STAGING_DIR)) || true
	mkdir -p $(STAGING_DIR)/node_modules
	cp $^ $(STAGING_DIR)
	mv $(STAGING_DIR)/$< $(STAGING_DIR)/index.mjs
	cd $(STAGING_DIR); npm install mathjs
	cd $(STAGING_DIR); $(ZIP) -r ../$@ *
