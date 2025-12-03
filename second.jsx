// App.js
import React, { useState, useEffect } from 'react';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    if (query.length < 2) {
      setResults([]);
      return;
    }
    const delayDebounce = setTimeout(() => {
      fetch(`http://localhost:8000/search?q=${encodeURIComponent(query)}`)
        .then(res => res.json())
        .then(data => setResults(data))
        .catch(err => console.error('API Error:', err));
    }, 300);

    return () => clearTimeout(delayDebounce);
  }, [query]);

  return (
    <div style={{ margin: '20px', fontFamily: 'Arial' }}>
      <h2>Ayush Disease Search (NAMASTE + ICD-11 TM2)</h2>
      <input
        type="text"
        style={{ width: '400px', padding: '8px', fontSize: '16px' }}
        placeholder="Enter disease description or code..."
        value={query}
        onChange={e => {
          setQuery(e.target.value);
          setSelected(null);
        }}
      />
      <ul style={{ listStyleType: 'none', paddingLeft: 0 }}>
        {results.map((item, idx) => (
          <li
            key={idx}
            onClick={() => setSelected(item)}
            style={{ cursor: 'pointer', padding: '6px', borderBottom: '1px solid #ccc' }}
          >
            <b>{item.NAMASTE_code}</b>: {item.NAMASTE_text}
          </li>
        ))}
      </ul>
      {selected && (
        <div style={{ marginTop: '20px', padding: '10px', border: '1px solid #888', borderRadius: '8px' }}>
          <h3>Selected NAMASTE Disease</h3>
          <p><b>Code:</b> {selected.NAMASTE_code}</p>
          <p><b>Description:</b> {selected.NAMASTE_text}</p>
          <h4>Mapped ICD-11 TM2 Disease</h4>
          <p><b>Code:</b> {selected.closest_ICD11_code}</p>
          <p><b>Name:</b> {selected.closest_ICD11_name}</p>
          <p><b>Similarity Score:</b> {selected.similarity_score.toFixed(3)}</p>
        </div>
      )}
    </div>
  );
}

export default App;
