const fs = require('fs');
const path = require('path');

const walk = function(dir) {
    let results = [];
    const list = fs.readdirSync(dir);
    list.forEach(function(file) {
        file = path.join(dir, file);
        const stat = fs.statSync(file);
        if (stat && stat.isDirectory()) { 
            results = results.concat(walk(file));
        } else { 
            if(file.endsWith('.jsx')) results.push(file);
        }
    });
    return results;
}

const files = walk(path.join(__dirname, 'frontend', 'src'));

const removeClasses = (classNameStr) => {
    // Classes to remove
    const toRemove = [
        /max-w-[a-z0-9\[\]]+/,
        /w-full/,
        /mx-auto/,
        /py-\d+/,
        /px-\d+/,
        /my-auto/,
        /my-\d+/,
        /space-y-\d+/ // "uneven paddings / margins"
    ];

    let classes = classNameStr.split(/\s+/);
    classes = classes.filter(c => {
        if (!c) return false;
        // Don't remove these from auth pages or generic text max-w if not on main/container. But let's check
        // Actually, let's just remove them.
        return !toRemove.some(regex => regex.test(c));
    });

    // Ensure cardio-container is present if it was there
    if (classNameStr.includes('cardio-container') && !classes.includes('cardio-container')) {
        classes.push('cardio-container');
    }
    return classes.join(' ');
};

files.forEach(file => {
    let content = fs.readFileSync(file, 'utf8');
    let modified = false;

    // 1. Standardize <main> tags
    content = content.replace(/<main\s+className=["']([^"']+)["']/g, (match, classNames) => {
        if (classNames.includes('auth-content-container')) {
            return match; // Leave auth pages alone as they have their own standard
        }
        
        let newClasses = removeClasses(classNames);
        
        // Ensure cardio-container and flex-1 are present for <main>
        const classesArray = newClasses.split(' ').filter(Boolean);
        if (!classesArray.includes('cardio-container')) classesArray.push('cardio-container');
        if (!classesArray.includes('flex-1')) classesArray.push('flex-1');
        
        modified = true;
        return `<main className="${classesArray.join(' ')}"`;
    });

    // 2. Standardize section and div tags that have cardio-container
    content = content.replace(/<(div|section)\s+className=["']([^"']*cardio-container[^"']*)["']/g, (match, tag, classNames) => {
        let newClasses = removeClasses(classNames);
        modified = true;
        return `<${tag} className="${newClasses}"`;
    });

    if (modified) {
        fs.writeFileSync(file, content, 'utf8');
        console.log(`Updated ${file}`);
    }
});
