AWS_REGION = us-east-1
STAGING_DIR = stage
ZIP = zip

AWS_NODE_MODULES = mathjs ajv
NODE_FILES = $(addprefix ../js/, cigartree.js likelihood.js mathutil.js tkf92.js validator.js)

lambda: evoldeeds-crud.lambda

MODEL = lg08evol.json

s3: frontend-client-test.s3.build
	cd ../frontend-client; aws s3 sync dist s3://evoldeeds/test

s3-clean: frontend-client-test.s3.build
	cd ../frontend-client; aws s3 sync --delete dist s3://evoldeeds/test

frontend-client-test.s3.build:
	cd ../frontend-client; PUBLIC_URL=https://api.evoldeeds.com/test npm run build

%.lambda: %.zip
	aws lambda update-function-code --region $(AWS_REGION) --publish --zip-file fileb://$< --function-name $* >$@
	rm -rf $*.zip index.mjs $(STAGING_DIR)

evoldeeds-crud.zip: api/evoldeeds-crud.js
	(test -e $(STAGING_DIR) && rm -rf $(STAGING_DIR)) || true
	mkdir -p $(STAGING_DIR)/node_modules
	cp $^ $(STAGING_DIR)
	mv $(STAGING_DIR)/$(notdir $<) $(STAGING_DIR)/index.mjs
	cp $(NODE_FILES) ../package.json $(STAGING_DIR)
	cp ../data/$(MODEL) $(STAGING_DIR)/model.json
	cd $(STAGING_DIR); npm install $(AWS_NODE_MODULES)
	cd $(STAGING_DIR); $(ZIP) -r ../$@ *
