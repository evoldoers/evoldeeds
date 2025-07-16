import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  DeleteCommand,
  GetCommand,
  ScanCommand,
} from "@aws-sdk/lib-dynamodb";

import Getopt from 'node-getopt';
import * as readline from 'readline';

const awsDefaultRegion = 'us-east-1';
const client = new DynamoDBClient({ region: process.env.AWS_DEFAULT_REGION || awsDefaultRegion });
const dynamo = DynamoDBDocumentClient.from(client);

// Set up command line options
const getopt = new Getopt([
    ['f', 'force', 'Do not ask for confirmation (default: false)'],
    ['k', 'keephistory', 'Keep history records associated with the family (default: false)'],
    ['i', 'ignorefamily', 'Ignore family (including existence check) and only delete history records (default: false)'],
    ['h', 'help', 'Display this help']
]);

const opt = getopt.parse(process.argv.slice(2));

if (opt.options.help) {
    console.log(getopt.getHelp());
    process.exit(0);
}

if (opt.argv.length != 1) {
    console.error('Usage: ' + process.argv[1] + ' [options] id');
    console.error(getopt.getHelp());
    process.exit(1);
}

if (opt.options.ignorefamily && opt.options.keephistory) {
    console.error('Cannot use --ignorefamily and --keephistory together');
    process.exit(1);
}

const familyTableName = "evoldeeds-families";
const historyTableName = "evoldeeds-histories";

const [ familyId ] = opt.argv;

const askConfirmation = (question) => {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    
    return new Promise((resolve) => {
        rl.question(question, (answer) => {
            rl.close();
            resolve(answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes');
        });
    });
};

const destroy = async (id) => {
    if (!opt.options.force) {
        const confirmed = await askConfirmation(`Are you sure you want to delete ${opt.options.ignorefamily ? `history records for family ${id}` : (opt.options.keephistory ? `family ${id}` : `family ${id} and its histories`)}? (y/n): `);
        if (!confirmed) {
            console.log('Operation cancelled.');
            return;
        }
    }
    
    try {
        if (!opt.options.ignorefamily) {
            // First check if the family exists
            const getResult = await dynamo.send(
                new GetCommand({
                    TableName: familyTableName,
                    Key: {
                        id
                    }
                })
            );
            
            if (!getResult.Item) {
                console.error(`Family '${id}' was not found in the database.`);
                process.exit(1);
            }
            
            // Delete the family
            await dynamo.send(
                new DeleteCommand({
                    TableName: familyTableName,
                    Key: {
                        id
                    }
                })
            );
            console.log(`Deleted family ${id}`);
        }
        
        // Delete history records if keephistory is not set
        if (!opt.options.keephistory) {
            await deleteHistoryRecords(id);
        }
        
    } catch (error) {
        console.error(`Error deleting family '${id}':`, error.message);
        if (error.name === 'ResourceNotFoundException') {
            console.error(`Table was not found.`);
        }
        process.exit(1);
    }
};

const deleteHistoryRecords = async (familyId) => {
    try {
        // Find all history records for this family
        const scanResult = await dynamo.send(
            new ScanCommand({
                TableName: historyTableName,
                FilterExpression: 'family_id = :familyId',
                ExpressionAttributeValues: {
                    ':familyId': familyId
                }
            })
        );
        
        if (scanResult.Items && scanResult.Items.length > 0) {
            // Delete each history record
            for (const item of scanResult.Items) {
                await dynamo.send(
                    new DeleteCommand({
                        TableName: historyTableName,
                        Key: {
                            family_id: item.family_id,
                            created: item.created
                        }
                    })
                );
            }
            console.log(`Deleted ${scanResult.Items.length} history records for family ${familyId}`);
        } else {
            console.log(`No history records found for family ${familyId}`);
        }
    } catch (error) {
        console.error(`Error deleting history records for family '${familyId}':`, error.message);
        process.exit(1);
    }
};

destroy (familyId);
