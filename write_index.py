content = '''<!DOCTYPE html>
<html>
<head>
    <title>Spam Mail Detection</title>
    <style>
        body {
            font-family: Arial;
            background-color: #f4f4f4;
            text-align: center;
            margin-top: 100px;
        }
        textarea {
            width: 400px;
            height: 120px;
        }
        button {
            padding: 10px 20px;
            margin-top: 10px;
            font-size: 16px;
        }
        h2 {
            color: #333;
        }
    </style>
</head>
<body>

    <h2>AI-Based Spam Mail Detection</h2>

    <form action="/predict" method="post">
        <textarea name="email" placeholder="Enter email text here..." required></textarea><br>
        <button type="submit">Check</button>
    </form>

    {% if prediction %}
        <h3>Result: {{ prediction }}</h3>
    {% endif %}

</body>
</html>
'''

p = 'backend/frontend/templates/index.html'
with open(p, 'w', encoding='utf-8') as f:
    f.write(content)
print('WROTE', p)
