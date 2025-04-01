% Load the .mat file
matData = load('C:\Users\Laptop Shop\Downloads\Tr0.mat');

% Extract signals and labels
labels = matData.Tr0.labels;   % Extract labels
signals = matData.Tr0.signals; % Extract signals

% Ensure labels are in string format (if they are categorical or numeric)
if iscategorical(labels)
    labels = cellstr(labels); % Convert categorical to cell array of strings
elseif isnumeric(labels)
    labels = arrayfun(@num2str, labels, 'UniformOutput', false); % Convert numbers to strings
end

% Convert labels to binary (PD = 1, NonPD = 0)
binary_labels = zeros(size(labels)); % Initialize array with zeros
for i = 1:length(labels)
    if strcmpi(labels{i}, 'PD') % Case-insensitive comparison
        binary_labels(i) = 1; % Assign 1 for PD
    else
        binary_labels(i) = 0; % Assign 0 for NonPD
    end
end

% Ensure signals are numeric (if they are in a cell array format)
if iscell(signals)
    signals = cell2mat(signals); % Convert cell to numeric matrix
end

% Open a CSV file to write
csvFile = 'C:\Users\Laptop Shop\Downloads\signals_labels_binary.csv';
fileID = fopen(csvFile, 'w');

% Write the header
fprintf(fileID, 'Signal Index,Signal Values,Label\n');

% Loop through each signal and write data to CSV
for i = 1:size(signals, 1) % Iterate over rows (signals)
    signal_values = sprintf('%f,', signals(i, :)); % Convert row to comma-separated string
    signal_values = signal_values(1:end-1); % Remove last comma
    fprintf(fileID, '%d,"%s",%d\n', i, signal_values, binary_labels(i)); % Write to file
end

% Close file
fclose(fileID);

fprintf('CSV file saved successfully at: %s\n', csvFile);
