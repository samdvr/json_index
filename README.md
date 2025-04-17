# JsonIndex

A simple document indexing and querying library built on top of Fjall, providing JSON document storage with flattened key-value indexing for efficient queries.

## Features

- **Document Storage**: Store JSON documents with unique IDs
- **Automatic Indexing**: Automatically flattens and indexes JSON documents for fast retrieval
- **Simple Querying**: Query documents using flattened key-value patterns
- **Complex Queries**: Perform complex AND/OR queries across multiple conditions
- **Document Management**: Add, query, and delete documents with ease

## Usage

```rust
use jsonindex::JsonIndexDB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open database
    let keyspace = fjall::Config::new("db_data").open()?;
    let json_index = JsonIndexDB::new(keyspace)?;

    // Index a document
    let doc = r#"{"name": "John Doe", "age": 30, "address": {"city": "Anytown"}}"#;
    let doc_id = "unique-id-123";
    json_index.index_document(doc, doc_id)?;

    // Simple query
    let results = json_index.query("name=John Doe")?;
    for (id, doc) in results {
        println!("Found document: {}", doc);
    }

    // Complex query (AND operation)
    let results = json_index.complex_query(&[
        "name=John Doe",
        "age=30",
    ], QueryLogic::And)?;

    // Complex query (OR operation)
    let results = json_index.complex_query(&[
        "address.city=Anytown",
        "address.city=Springfield",
    ], QueryLogic::Or)?;

    // Delete a document
    json_index.delete_document(doc_id)?;

    Ok(())
}
```

## How It Works

JsonIndex flattens nested JSON documents into key-value pairs with paths representing the hierarchy. For example:

```json
{
  "name": "John",
  "address": {
    "city": "Anytown"
  }
}
```

Gets indexed as:

- `name=John`
- `address.city=Anytown`

Each indexed key-value pair points to a list of document IDs that contain that key-value pair, enabling efficient lookups.

## License

MIT
